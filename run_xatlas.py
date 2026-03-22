#!/usr/bin/env python3
"""
run_xatlas.py
Read export_data.json produced by the addon, run xatlas to produce charts,
then derive seam edges (original edge indices) and write import_seams.json.
"""

import json
import os
import sys
from collections import defaultdict, Counter

ANGLE_THRESHOLD_DEG = 70.0  # treat edges with dihedral angle above this as seams, even if xatlas put them in same chart
STRESS_THRESHOLD = 2000
LOG_STRESS_DIFF = 4.0
EDGE_STRESS_MIN = 2.0
EPS = 1e-9
CHART_MIN_TRIS = 3
DIAGNOSTIC_TOP_N = 12


try:
    import xatlas
except Exception as exc:
    print("Failed to import xatlas:", exc)
    print("Install with: pip install xatlas")
    sys.exit(1)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fan_triangulate(face_verts):
    # face_verts: list of vertex indices (may be len>=3)
    # return list of (a,b,c) triangle triples using a simple fan
    if len(face_verts) < 3:
        return []
    if len(face_verts) == 3:
        return [tuple(face_verts)]
    a = face_verts[0]
    tris = []
    for i in range(1, len(face_verts)-1):
        tris.append((a, face_verts[i], face_verts[i+1]))
    return tris

def build_mesh_from_export(export):
    vertices = [v['co'] for v in export['vertices']]
    triangles = []
    tri_to_face = []
    if 'triangles' in export and len(export['triangles'])>0:
        for t in export['triangles']:
            triangles.append(list(t['verts']))
            tri_to_face.append(int(t.get('orig_face', -1)))
    else:
        # fallback to fan triangulation of faces
        for f in export['faces']:
            tri_list = fan_triangulate(f['verts'])
            for tri in tri_list:
                triangles.append(list(tri))
                tri_to_face.append(f['index'])
    return vertices, triangles, tri_to_face

import math as _math

def tri_area_3d(vertices, tri):
    # vertices: list of [x,y,z], tri: (i0,i1,i2)
    a = vertices[tri[0]]
    b = vertices[tri[1]]
    c = vertices[tri[2]]
    ax,ay,az = a; bx,by,bz = b; cx,cy,cz = c
    ux,uy,uz = bx-ax, by-ay, bz-az
    vx,vy,vz = cx-ax, cy-ay, cz-az
    cxp = uy*vz - uz*vy
    cyp = uz*vx - ux*vz
    czp = ux*vy - uy*vx
    return 0.5 * _math.sqrt(cxp*cxp + cyp*cyp + czp*czp)

def tri_area_2d(uvs, tri):
    # uvs: list of (u,v) for triangle's vertices, tri: (i0,i1,i2) indices into uvs
    (u0,v0),(u1,v1),(u2,v2) = uvs[tri[0]], uvs[tri[1]], uvs[tri[2]]
    return abs(0.5 * ((u1-u0)*(v2-v0) - (u2-u0)*(v1-v0)))


def run_xatlas_and_get_chart_per_triangle(vertices, triangles):
    # Try Atlas API first
    try:
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, triangles)
        atlas.generate()

        atlas_vmapping = atlas_indices = atlas_uvs = None

        try:
            atlas_vmapping, atlas_indices, atlas_uvs = atlas[0]
        except Exception:

            try:
                atlas_vmapping, atlas_indices, atlas_uvs = xatlas.parametrize(vertices, triangles)
            except Exception:
                atlas_vmapping, atlas_indices, atlas_uvs = None, None, None

        if atlas_vmapping is not None:
            print("Info: atlas UV output available (using atlas or parametrize).")
        else:
            atlas_vmapping = list(map(int, atlas_vmapping))
            atlas_indices = [tuple(map(int, t)) for t in atlas_indices.reshape(-1,3)] if hasattr(atlas_indices, 'reshape') else list(map(tuple, atlas_indices))
            atlas_uvs = [tuple(map(float, u)) for u in atlas_uvs]

        if atlas_vmapping is not None and atlas_indices is not None:
            vm_to_tri = defaultdict(list)
            for ti, tri in enumerate(atlas_indices):
                for v in tri:
                    vm_to_tri[int(v)].append(ti)
            adj = [[] for _ in range(len(atlas_indices))]
            for tri_list in vm_to_tri.values():
                for i in tri_list:
                    for j in tri_list:
                        if i != j:
                            adj[i].append(j)
            seen = [False]*len(atlas_indices)
            out_tri_chart = [-1]*len(atlas_indices)
            cur = 0
            for i in range(len(atlas_indices)):
                if seen[i]:
                    continue
                stack = [i]; seen[i] = True
                while stack:
                    a = stack.pop()
                    out_tri_chart[a] = cur
                    for b in adj[a]:
                        if not seen[b]:
                            seen[b] = True
                            stack.append(b)
                cur += 1

            # Map output-triangles back to input triangles by original vertex indices
            mapping = defaultdict(list)
            for out_ti, out_tri in enumerate(atlas_indices):
                orig_triple = tuple(int(atlas_vmapping[idx]) for idx in out_tri)
                key = tuple(sorted(orig_triple))
                mapping[key].append(out_ti)

            # For each input triangle choose the majority chart among its mapped output triangles
            tri_chart_ids = []
            for tri in triangles:
                key = tuple(sorted(tri))
                out_tris = mapping.get(key, [])
                if out_tris:
                    c = Counter(out_tri_chart[t] for t in out_tris).most_common(1)[0][0]
                    tri_chart_ids.append(int(c))
                else:
                    tri_chart_ids.append(0)

        return tri_chart_ids, (atlas_vmapping, atlas_indices, atlas_uvs)
    except Exception:
        print("Atlas() path failed:", exc)

        # Fallback to high-level parametrize helper
    try:
        vmapping, out_indices, uvs = xatlas.parametrize(vertices, triangles)
        vmapping = list(map(int, vmapping))
        out_indices = [tuple(map(int, t)) for t in out_indices.reshape(-1,3)] if hasattr(out_indices, 'reshape') else list(map(tuple, out_indices))
        uvs = [tuple(map(float, u)) for u in uvs]

        vm_to_tri = defaultdict(list)
        for ti, tri in enumerate(out_indices):
            for v in tri:
                vm_to_tri[v].append(ti)
        adj = [[] for _ in range(len(out_indices))]
        for tri_list in vm_to_tri.values():
            for i in tri_list:
                for j in tri_list:
                    if i != j:
                        adj[i].append(j)
        seen = [False]*len(out_indices)
        out_tri_chart = [-1]*len(out_indices)
        cur = 0
        for i in range(len(out_indices)):
            if seen[i]:
                continue
            stack=[i]; seen[i]=True
            while stack:
                a = stack.pop()
                out_tri_chart[a]=cur
                for b in adj[a]:
                    if not seen[b]:
                        seen[b]=True
                        stack.append(b)
            cur += 1

        mapping = defaultdict(list)
        for out_ti, out_tri in enumerate(out_indices):
            orig_triple = tuple(int(vmapping[idx]) for idx in out_tri)
            key = tuple(sorted(orig_triple))
            mapping[key].append(out_ti)

        tri_chart_ids = []
        for tri in triangles:
            key = tuple(sorted(tri))
            out_tris = mapping.get(key, [])
            if out_tris:
                c = Counter(out_tri_chart[t] for t in out_tris).most_common(1)[0][0]
                tri_chart_ids.append(int(c))
            else:
                tri_chart_ids.append(0)

        return tri_chart_ids, (vmapping, out_indices, uvs)
        
    except Exception as exc:
            print("xatlas fallback failed:", exc)
            return None, (None, None, None)






# def derive_seams(export, triangles, tri_to_face, tri_chart_ids):
#     # Map triangle chart id -> original face index groups
#     face_to_tri_charts = defaultdict(list)
#     for tri_idx, face_idx in enumerate(tri_to_face):
#         face_to_tri_charts[face_idx].append(tri_chart_ids[tri_idx])
#     # Representative face chart: majority of its triangles' charts
#     face_chart = {}
#     for face_idx, charts in face_to_tri_charts.items():
#         face_chart[face_idx] = Counter(charts).most_common(1)[0][0]
#     # For each original edge from export['edges'], check adjacent faces
#     seams = []
#     for e in export['edges']:
#         edge_idx = e['index']
#         # Need to find adjacent faces that contain this edge (two faces or 1 if boundary)
#         v0, v1 = e['verts']
#         adjacent_faces = []
#         for f in export['faces']:
#             verts = f['verts']
#             # if both vertices present in face, consider it adjacent
#             if v0 in verts and v1 in verts:
#                 adjacent_faces.append(f['index'])
#         if len(adjacent_faces) == 2:
#             fa, fb = adjacent_faces
#             ca = face_chart.get(fa, None)
#             cb = face_chart.get(fb, None)
#             if ca is None or cb is None:
#                 continue
#             if ca != cb:
#                 seams.append(edge_idx)
#         else:
#             # boundary edge: usually set seam
#             seams.append(edge_idx)
#     # dedupe and sort
#     seams = sorted(set(seams))
#     return seams


def derive_seams(export, triangles, tri_to_face, tri_chart_ids,
                 triangle_area_ratio=None, triangle_log_ratio=None, chart_sizes=None):

    if triangle_area_ratio is None:
        triangle_area_ratio = [1.0]*len(triangles)
    if triangle_log_ratio is None:
        triangle_log_ratio = [0.0]*len(triangles)

    # Build edge -> triangle index map from triangulated triangles
    edge_to_tris = {}
    edge_idx_to_verts = {}
    for ti, tri in enumerate(triangles):
        verts = tri
        edges = [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]
        for a,b in edges:
            key = tuple(sorted((int(a), int(b))))
            edge_to_tris.setdefault(key, []).append(ti)
    for e in export.get('edges', []):
        edge_idx_to_verts[int(e['index'])] = (int(e['verts'][0]), int(e['verts'][1]))

    seams = []
    stats = {"angle":0, "chart":0, "chart_ignored":0, "stress":0, "boundary":0, "degenerate":0, "post_removed":0}

    for e in export['edges']:
        edge_idx = e['index']
        key = tuple(sorted((int(e['verts'][0]), int(e['verts'][1]))))
        tris = edge_to_tris.get(key, [])

        # 1) dihedral rule

        try: 
            edge_angle = float(e.get('angle', 0.0))
        except Exception:
            edge_angle = 0.0
        if ANGLE_THRESHOLD_DEG and edge_angle >= float(ANGLE_THRESHOLD_DEG):
            seams.append(edge_idx); stats["angle"] += 1; continue



        # 2) nothing maps to triangles -> seam (degenerate)
        if len(tris) == 0:
            seams.append(edge_idx); stats["degenerate"] += 1; continue


        # 3) chart boundary rule
        charts = set(tri_chart_ids[t] for t in tris if 0 <= t < len(tri_chart_ids))
        if len(charts) > 1:
            # ignore tiny charts (likely fragmentation noise)
            if chart_sizes is not None:
                sizes = [chart_sizes.get(c, 0) for c in charts]
                small = min(sizes) if sizes else 0
                if small < CHART_MIN_TRIS:
                    stats["chart_ignored"] += 1
                    continue
            seams.append(edge_idx); stats["chart"] += 1; continue

        # 4) stress-based rule (per-edge relative test)

        # gather ratios/logs for triangles adjacent to this edge
        ratios = [triangle_area_ratio[t] for t in tris if 0 <= t < len(triangle_area_ratio)]
        logs = [triangle_log_ratio[t] for t in tris if 0 <= t < len(triangle_log_ratio)]
        max_ratio = max(ratios) if ratios else 0.0
        log_diff = 0.0
        if len(logs) >= 2:
            log_diff = max(abs(logs[i]-logs[j]) for i in range(len(logs)) for j in range(i+1, len(logs)))

        # If edge has two triangles, require BOTH triangles to be reasonably stressed
        if len(tris) == 2:
            if min(ratios) >= EDGE_STRESS_MIN and max_ratio >= STRESS_THRESHOLD and log_diff >= LOG_STRESS_DIFF:
                seams.append(edge_idx); stats["stress"] += 1; continue
        else:
            # for single-triangle (boundary-like) or other cases, keep previous behavior:
            if max_ratio >= STRESS_THRESHOLD and log_diff >= LOG_STRESS_DIFF:
                seams.append(edge_idx); stats["stress"] += 1; continue

        # 5) boundary-like single-triangle edge -> seam
        if len(tris) == 1:
            seams.append(edge_idx); stats["boundary"] += 1

    seams_set = set(seams)
    vert_seam_degree = defaultdict(int)
    for s in seams_set:
        verts = edge_idx_to_verts.get(s)
        if verts:
            vert_seam_degree[verts[0]] += 1
            vert_seam_degree[verts[1]] += 1

    removed = []
    for s in list(seams_set):
        verts = edge_idx_to_verts.get(s)
        if not verts:
            continue
        a,b = verts
        if vert_seam_degree.get(a,0) == 1 and vert_seam_degree.get(b,0) == 1:
            seams_set.remove(s)
            stats["post_removed"] += 1
            # decrement degrees (keeps counts consistent)
            vert_seam_degree[a] -= 1
            vert_seam_degree[b] -= 1


    seams = sorted(set(seams))
    print("Seam rule counts:", stats)
    return seams

def main():
    if len(sys.argv) < 2:
        print("Usage: run_xatlas.py /path/to/export_data.json")
        sys.exit(1)
    json_path = sys.argv[1]
    export = load_json(json_path)
    vertices, triangles, tri_to_face = build_mesh_from_export(export)
    print(f"Built {len(vertices)} vertices, {len(triangles)} triangles (from {len(export['faces'])} faces)")
    tri_chart_ids, atlas_out = run_xatlas_and_get_chart_per_triangle(vertices, triangles)
    if tri_chart_ids is None:
        print("Failed to compute charts with xatlas.")
        sys.exit(2)

    chart_sizes = Counter(tri_chart_ids)
    print("Chart count:", len(chart_sizes))
    print("Chart sizes (top 12):", chart_sizes.most_common(12))

        # attempt to compute per-input-triangle area_ratio using atlas output if available
    atlas_vmapping, atlas_indices, atlas_uvs = atlas_out
    triangle_area_ratio = [1.0] * len(triangles)  # default safe value
    triangle_log_ratio = [0.0] * len(triangles)
    if atlas_vmapping is not None and atlas_indices is not None and atlas_uvs is not None:
        # Build map: sorted(orig_vertex_tuple) -> list of output-triangle indices
        orig_uvs = defaultdict(list)
        for out_vidx, orig_vidx in enumerate(atlas_vmapping):
            try:
                uv = tuple(map(float, atlas_uvs[out_vidx]))
                orig_uvs[int(orig_vidx)].append(uv)
            except Exception:
                continue

        # Helper: average aggregated UVs for an original vertex (or return None)
        def avg_uv(uv_list):
            if not uv_list:
                return None
            ux = sum(u for u,v in uv_list)/len(uv_list)
            vx = sum(v for u,v in uv_list)/len(uv_list)
            return (ux, vx)

        degenerate_uv_count = 0
        multi_map_count = 0

    for i, in_tri in enumerate(triangles):

        a3 = vertices[in_tri[0]]; 
        b3 = vertices[in_tri[1]];
        c3 = vertices[in_tri[2]];

        ua = avg_uv(orig_uvs.get(in_tri[0], []))
        ub = avg_uv(orig_uvs.get(in_tri[1], []))
        uc = avg_uv(orig_uvs.get(in_tri[2], []))

        if ua is None or ub is None or uc is None:
            triangle_area_ratio[i] = 1.0
            triangle_log_ratio[i] = 0.0
            continue


        # Build 2x2 UV matrix B = [ (ub-ua)  (uc-ua) ] as columns
        b00 = ub[0] - ua[0]; b10 = ub[1] - ua[1]
        b01 = uc[0] - ua[0]; b11 = uc[1] - ua[1]
        detB = b00 * b11 - b01 * b10
        if abs(detB) <= EPS:
            triangle_area_ratio[i] = 1.0
            triangle_log_ratio[i] = 0.0
            degenerate_uv_count += 1
            continue
        invDet = 1.0 / detB
        invB00 =  b11 * invDet
        invB01 = -b01 * invDet
        invB10 = -b10 * invDet
        invB11 =  b00 * invDet


        a0x = b3[0] - a3[0]; a0y = b3[1] - a3[1]; a0z = b3[2] - a3[2]
        a1x = c3[0] - a3[0]; a1y = c3[1] - a3[1]; a1z = c3[2] - a3[2]


        # J = A * invB  (3x2)
        J00 = a0x * invB00 + a1x * invB10
        J10 = a0y * invB00 + a1y * invB10
        J20 = a0z * invB00 + a1z * invB10

        J01 = a0x * invB01 + a1x * invB11
        J11 = a0y * invB01 + a1y * invB11
        J21 = a0z * invB01 + a1z * invB11

        # 2x2 symmetric C = J^T * J
        C00 = J00*J00 + J10*J10 + J20*J20
        C01 = J00*J01 + J10*J11 + J20*J21
        C11 = J01*J01 + J11*J11 + J21*J21


        trace = C00 + C11
        disc = trace*trace - 4.0*(C00*C11 - C01*C01)
        disc = max(disc, 0.0)
        ev_high = 0.5*(trace + _math.sqrt(disc))
        ev_low  = 0.5*(trace - _math.sqrt(disc))


        s_high = _math.sqrt(max(ev_high, EPS))
        s_low  = _math.sqrt(max(ev_low, EPS))



        if s_low <= EPS:
            ratio = 1.0
        else:
            ratio = s_high / s_low


        ratio = min(ratio, 1e6)

        triangle_area_ratio[i] = ratio
        triangle_log_ratio[i] = _math.log(ratio + EPS)

    try:
        out_dir = os.path.dirname(json_path)
        stress_items = [
            {"tri_idx": i, "stress": float(triangle_area_ratio[i]),
             "orig_face": int(tri_to_face[i]) if i < len(tri_to_face) else -1}
            for i in range(len(triangle_area_ratio))
        ]
        stress_report = {
            "triangle_stress": stress_items,
            "tri_chart_ids": tri_chart_ids if 'tri_chart_ids' in locals() else None,
        }
        stress_path = os.path.join(out_dir, "stress_report.json")
        with open(stress_path, "w", encoding="utf-8") as sf:
            json.dump(stress_report, sf, indent=2)
        print(f"Wrote stress report to {stress_path}")
    except Exception as _e:
        print("Warning: failed to write stress_report.json:", _e)


    if atlas_vmapping is not None and atlas_indices is not None and atlas_uvs is not None:
        if 'degenerate_uv_count' in locals() and degenerate_uv_count:
            print(f"Warning: {degenerate_uv_count} degenerate UV triangles ignored for stress.")
        if 'multi_map_count' in locals() and multi_map_count:
            print(f"Info: {multi_map_count} input triangles map to multiple atlas output triangles (aggregated).")
    else:
        print("Atlas UV output unavailable; stress-based rules will use defaults.")



    if 'degenerate_uv_count' in locals() and degenerate_uv_count:
        print(f"Warning: {degenerate_uv_count} degenerate UV triangles ignored for stress.")
    if 'multi_map_count' in locals() and multi_map_count:
        print(f"Info: {multi_map_count} input triangles map to multiple atlas output triangles (aggregated).")
    if not (atlas_vmapping is not None and atlas_indices is not None and atlas_uvs is not None):
        print("Atlas UV output unavailable; stress-based rules will use defaults.")
    

    top = sorted(enumerate(triangle_area_ratio), key=lambda x: x[1], reverse=True)[:DIAGNOSTIC_TOP_N]
    print("Top stressed triangles (tri_idx, stress, orig_face):")
    for ti, val in top:
        face_idx = tri_to_face[ti] if ti < len(tri_to_face) else -1
        print(f"  {ti}: {val:.3g} (face {face_idx})")

    
    seams = derive_seams(export, triangles, tri_to_face, tri_chart_ids,
                         triangle_area_ratio, triangle_log_ratio, chart_sizes=chart_sizes)
    out_dir = os.path.dirname(json_path)
    out_path = os.path.join(out_dir, "import_seams.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(seams, f, indent=2)
    print(f"Wrote {len(seams)} seams to {out_path}")

if __name__ == "__main__":
    main()