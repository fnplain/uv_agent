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

def run_xatlas_and_get_chart_per_triangle(vertices, triangles):
    # Try Atlas API first
    try:
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, triangles)
        atlas.generate()
        # atlas.get_mesh_vertex_assignement(mesh_idx) -> (atlas_index_arr, chart_index_arr)
        atlas_index_arr, chart_index_arr = atlas.get_mesh_vertex_assignement(0)
        # chart_index_arr length == len(vertices) (assignment per original vertex)
        # For each input triangle (vertex indices), get chart id by majority of its vertices
        tri_chart_ids = []
        for tri in triangles:
            charts = [chart_index_arr[v] for v in tri]
            # choose majority, else first
            c = Counter(charts).most_common(1)[0][0]
            tri_chart_ids.append(int(c))
        return tri_chart_ids
    except Exception:
        # Fallback to high-level parametrize helper
        try:
            vmapping, out_indices, uvs = xatlas.parametrize(vertices, triangles)
            # out_indices is an array of triangles referencing vmapping indices
            # vmapping maps new vertices -> original vertex index
            # We'll compute per-output-triangle chart id by grouping triangles whose vmapping vertices share a chart id
            # But parametrize helper doesn't return chart ids directly, so we fall back to using connected UV islands.
            # Simpler approach: build UV adjacency and connected components to get chart ids.
            import numpy as np
            vmapping = list(map(int, vmapping))
            out_indices = [tuple(map(int, t)) for t in out_indices.reshape(-1,3)] if hasattr(out_indices, 'reshape') else out_indices
            uvs = [tuple(map(float, u)) for u in uvs]
            # Build graph of triangle adjacency via shared uv vertices (vmapping index)
            # Map vmapping index -> list of output triangle indices
            vm_to_tri = defaultdict(list)
            for ti, tri in enumerate(out_indices):
                for v in tri:
                    vm_to_tri[v].append(ti)
            # build adjacency and flood fill for components
            adj = [[] for _ in range(len(out_indices))]
            for tri_indices in vm_to_tri.values():
                for i in tri_indices:
                    for j in tri_indices:
                        if i != j:
                            adj[i].append(j)
            # flood fill
            seen = [False]*len(out_indices)
            tri_chart_ids = [-1]*len(out_indices)
            cur = 0
            for i in range(len(out_indices)):
                if seen[i]:
                    continue
                stack=[i]; seen[i]=True
                while stack:
                    a = stack.pop()
                    tri_chart_ids[a]=cur
                    for b in adj[a]:
                        if not seen[b]:
                            seen[b]=True
                            stack.append(b)
                cur += 1
            # tri_chart_ids correspond to the output triangles; we need to map them back to input triangles.
            # Assume output triangle order corresponds to input triangles (often true for parametrize helper).
            # We'll return tri_chart_ids[:len(triangles)] or pad as needed.
            if len(tri_chart_ids) >= len(triangles):
                return tri_chart_ids[:len(triangles)]
            else:
                # fallback: assign chart 0 for all
                return [0]*len(triangles)
        except Exception as exc:
            print("xatlas fallback failed:", exc)
            return None

def derive_seams(export, triangles, tri_to_face, tri_chart_ids):
    # Map triangle chart id -> original face index groups
    face_to_tri_charts = defaultdict(list)
    for tri_idx, face_idx in enumerate(tri_to_face):
        face_to_tri_charts[face_idx].append(tri_chart_ids[tri_idx])
    # Representative face chart: majority of its triangles' charts
    face_chart = {}
    for face_idx, charts in face_to_tri_charts.items():
        face_chart[face_idx] = Counter(charts).most_common(1)[0][0]
    # For each original edge from export['edges'], check adjacent faces
    seams = []
    for e in export['edges']:
        edge_idx = e['index']
        # Need to find adjacent faces that contain this edge (two faces or 1 if boundary)
        v0, v1 = e['verts']
        adjacent_faces = []
        for f in export['faces']:
            verts = f['verts']
            # if both vertices present in face, consider it adjacent
            if v0 in verts and v1 in verts:
                adjacent_faces.append(f['index'])
        if len(adjacent_faces) == 2:
            fa, fb = adjacent_faces
            ca = face_chart.get(fa, None)
            cb = face_chart.get(fb, None)
            if ca is None or cb is None:
                continue
            if ca != cb:
                seams.append(edge_idx)
        else:
            # boundary edge: usually set seam
            seams.append(edge_idx)
    # dedupe and sort
    seams = sorted(set(seams))
    return seams

def main():
    if len(sys.argv) < 2:
        print("Usage: run_xatlas.py /path/to/export_data.json")
        sys.exit(1)
    json_path = sys.argv[1]
    export = load_json(json_path)
    vertices, triangles, tri_to_face = build_mesh_from_export(export)
    print(f"Built {len(vertices)} vertices, {len(triangles)} triangles (from {len(export['faces'])} faces)")
    tri_chart_ids = run_xatlas_and_get_chart_per_triangle(vertices, triangles)
    if tri_chart_ids is None:
        print("Failed to compute charts with xatlas.")
        sys.exit(2)
    seams = derive_seams(export, triangles, tri_to_face, tri_chart_ids)
    out_dir = os.path.dirname(json_path)
    out_path = os.path.join(out_dir, "import_seams.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(seams, f, indent=2)
    print(f"Wrote {len(seams)} seams to {out_path}")

if __name__ == "__main__":
    main()