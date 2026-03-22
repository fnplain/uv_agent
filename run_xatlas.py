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
STRESS_THRESHOLD = 5.0
LOG_STRESS_DIFF = 1.5
EDGE_STRESS_MIN = 1.5
EPS = 1e-9

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
                 triangle_area_ratio=None, triangle_log_ratio=None):

    if triangle_area_ratio is None:
        triangle_area_ratio = [1.0]*len(triangles)
    if triangle_log_ratio is None:
        triangle_log_ratio = [0.0]*len(triangles)

    # Build edge -> triangle index map from triangulated triangles
    edge_to_tris = {}
    for ti, tri in enumerate(triangles):
        verts = tri
        edges = [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]
        for a,b in edges:
            key = tuple(sorted((int(a), int(b))))
            edge_to_tris.setdefault(key, []).append(ti)

    seams = []
    stats = {"angle":0, "chart":0, "stress":0, "boundary":0, "degenerate":0}

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

        def dist3(p,q):
            return _math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)
        L3 = [dist3(a3,b3), dist3(b3,c3), dist3(c3,a3)]

        ua = avg_uv(orig_uvs.get(in_tri[0], []))
        ub = avg_uv(orig_uvs.get(in_tri[1], []))
        uc = avg_uv(orig_uvs.get(in_tri[2], []))

        if ua is None or ub is None or uc is None:
            triangle_area_ratio[i] = 1.0
            triangle_log_ratio[i] = 0.0
            continue

        def dist2(p,q):
            return _math.hypot(p[0]-q[0], p[1]-q[1])
        L2 = [dist2(ua,ub), dist2(ub,uc), dist2(uc,ua)]

        if any(l <= EPS for l in L2):
            triangle_area_ratio[i] = 1.0
            triangle_log_ratio[i] = 0.0
            degenerate_uv_count += 1
            continue

        # per-edge 3D/2D ratios
        ratios = [ (L3[j] / L2[j]) for j in range(3) ]
        # triangle stress metric: ratio_spread = max/min
        ratio_min = max(min(ratios), 1e-12)
        ratio_max = max(ratios)
        ratio_spread = ratio_max / ratio_min if ratio_min>0 else ratio_max
        # store as "area_ratio" variable used elsewhere (and log)
        triangle_area_ratio[i] = min(ratio_spread, 1e6)
        triangle_log_ratio[i] = _math.log(triangle_area_ratio[i] + EPS)

    if atlas_vmapping is not None and atlas_indices is not None and atlas_uvs is not None:
        if 'degenerate_uv_count' in locals() and degenerate_uv_count:
            print(f"Warning: {degenerate_uv_count} degenerate UV triangles ignored for stress.")
        if 'multi_map_count' in locals() and multi_map_count:
            print(f"Info: {multi_map_count} input triangles map to multiple atlas output triangles (aggregated).")
    else:
        print("Atlas UV output unavailable; stress-based rules will use defaults.")


    
    seams = derive_seams(export, triangles, tri_to_face, tri_chart_ids,
                         triangle_area_ratio, triangle_log_ratio)
    out_dir = os.path.dirname(json_path)
    out_path = os.path.join(out_dir, "import_seams.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(seams, f, indent=2)
    print(f"Wrote {len(seams)} seams to {out_path}")

if __name__ == "__main__":
    main()