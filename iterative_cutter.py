from collections import defaultdict, deque, Counter
import heapq
from typing import List, Tuple, Set, Dict
import run_xatlas
import math as _math
import json
import os

def build_edge_to_tris(triangles: List[List[int]]):
    edge_to_tris = defaultdict(list)
    for ti, tri in enumerate(triangles):
        a,b,c = tri
        for u,v in ((a,b),(b,c),(c,a)):
            key = tuple(sorted((int(u),int(v))))
            edge_to_tris[key].append(ti)
    return dict(edge_to_tris)

def build_tri_adjacency(triangles: List[List[int]]):
    edge_to_tris = build_edge_to_tris(triangles)
    tri_neighbors = [set() for _ in range(len(triangles))]
    shared_edge_between = {}
    for edge, tris in edge_to_tris.items():
        for i in range(len(tris)):
            for j in range(i+1, len(tris)):
                ta, tb = tris[i], tris[j]
                tri_neighbors[ta].add(tb)
                tri_neighbors[tb].add(ta)
                shared_edge_between[(ta,tb)] = edge
                shared_edge_between[(tb,ta)] = edge
    return tri_neighbors, shared_edge_between, edge_to_tris

def grow_region(seed_tri: int, tri_neighbors: List[Set[int]], min_tris: int=20, max_tris: int=800,
                stop_at_seams: bool=True, seam_edge_keys: Set[Tuple[int,int]]=None):
    """
    BFS expand from seed_tri until region size >= min_tris or max_tris reached.
    If stop_at_seams and seam_edge_keys provided, expansion will not cross triangles
    that would cross a seam edge (i.e., shared edge in seam_edge_keys).
    Returns set of triangle indices in region.
    """
    if seam_edge_keys is None:
        seam_edge_keys = set()

    region = set([seed_tri])
    q = deque([seed_tri])
    while q and len(region) < max_tris:
        t = q.popleft()
        for n in tri_neighbors[t]:
            if n in region:
                continue

            shared = None

            region.add(n)
            q.append(n)
            if len(region) >= min_tris and len(region) >= min_tris:
                # early exit when we've reached min_tris (we'll still respect max_tris)
                if len(region) >= min_tris:
                    # continue to allow slight overshoot until queue drains a bit
                    pass
        if len(region) >= min_tris:
            break
    # trim if beyond max_tris
    if len(region) > max_tris:
        region = set(list(region)[:max_tris])
    return region

def dijkstra_to_nearest_boundary(seed_tri: int, triangles: List[List[int]], vertices: List[Tuple[float,float,float]],
                                 tri_neighbors: List[Set[int]], shared_edge_between: Dict[Tuple[int,int], Tuple[int,int]],
                                 edge_to_tris: Dict[Tuple[int,int], List[int]],
                                 seam_edges: Set[Tuple[int,int]]=None,
                                 region_mask: Set[int]=None,
                                 max_search: int=5000):
    """
    Dijkstra on triangle dual graph from seed_tri to nearest triangle that touches a mesh boundary
    or an existing seam. Edge weight is length of the shared edge (geometric).
    Returns list of triangle indices forming the path (inclusive), or [] if not found.
    If region_mask is provided, search is constrained to that set.
    """
    if seam_edges is None:
        seam_edges = set()
    N = len(triangles)
    seen = {}
    pq = []
    heapq.heappush(pq, (0.0, seed_tri, -1))  # (cost, tri, prev)
    prev = {seed_tri: None}
    cost_so_far = {seed_tri: 0.0}
    found_tri = None
    iterations = 0

    while pq and iterations < max_search:
        iterations += 1
        cost, t, p = heapq.heappop(pq)
        if seen.get(t):
            continue
        seen[t] = True
        prev[t] = p

        # Check if triangle t touches a boundary edge (edge_to_tris len==1) or an existing seam
        tri_verts = triangles[t]
        for a,b in [(tri_verts[0],tri_verts[1]),(tri_verts[1],tri_verts[2]),(tri_verts[2],tri_verts[0])]:
            key = tuple(sorted((int(a),int(b))))
            tris = edge_to_tris.get(key, [])
            # boundary
            if len(tris) == 1:
                found_tri = t
                break
            # existing seam provided as vertex-pair keys
            if key in seam_edges:
                found_tri = t
                break
        if found_tri is not None:
            break

        for nbr in tri_neighbors[t]:
            if region_mask is not None and nbr not in region_mask:
                continue
            if seen.get(nbr):
                continue
            edge = shared_edge_between.get((t,nbr))
            if edge is None:
                w = 1.0
            else:
                # compute geometric length as weight
                p0 = vertices[edge[0]]; p1 = vertices[edge[1]]
                dx = p0[0]-p1[0]; dy = p0[1]-p1[1]; dz = p0[2]-p1[2]
                w = (dx*dx + dy*dy + dz*dz)**0.5
            new_cost = cost + w
            prev_cost = cost_so_far.get(nbr, float('inf'))
            if new_cost < prev_cost:
                cost_so_far[nbr] = new_cost
                heapq.heappush(pq, (new_cost, nbr, t))
                prev[nbr] = t

    if found_tri is None:
        return []

    # reconstruct path
    path = []
    cur = found_tri
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path

def triangles_to_edge_path(path_tris: List[int], shared_edge_between: Dict[Tuple[int,int], Tuple[int,int]]):
    """
    Convert a triangle-triangle path to a list of mesh edges (vertex-pair keys)
    that connect the sequence (the shared edges between consecutive triangles).
    """
    edges = []
    for i in range(len(path_tris)-1):
        a = path_tris[i]; b = path_tris[i+1]
        key = shared_edge_between.get((a,b))
        if key is None:
            continue
        edges.append(tuple(sorted(key)))
    return edges

def extract_region_submesh(vertices: List[Tuple[float,float,float]],
                           triangles: List[List[int]],
                           region_tris: Set[int]):
    """
    Build a standalone submesh for region_tris:
    - returns (sub_vertices, sub_triangles, sub_tri_to_orig_tri, sub_vert_to_orig_vert)
      where sub_tri indices refer to sub_vertices.
    This is suitable to run xatlas on the isolated region (region boundary becomes mesh border).
    """
    used_verts = set()
    for ti in region_tris:
        for v in triangles[ti]:
            used_verts.add(int(v))
    # map original vert -> sub vert index
    orig_to_sub = {}
    sub_vertices = []
    for i, ov in enumerate(sorted(used_verts)):
        orig_to_sub[ov] = i
        sub_vertices.append(vertices[ov])
    sub_triangles = []
    sub_tri_to_orig = []
    for ti in sorted(region_tris):
        tri = triangles[ti]
        sub_tri = [orig_to_sub[int(tri[0])], orig_to_sub[int(tri[1])], orig_to_sub[int(tri[2])]]
        sub_triangles.append(sub_tri)
        sub_tri_to_orig.append(ti)
    # inverse map: sub_idx -> orig idx
    sub_vert_to_orig = {sub_idx: orig for orig, sub_idx in orig_to_sub.items()}
    return sub_vertices, sub_triangles, sub_tri_to_orig, sub_vert_to_orig

# Example high-level helper (not executed here): pick top stressed triangle, grow a region, extract submesh
def pick_and_extract_top_region(triangle_stress: List[float], triangles: List[List[int]], vertices: List[Tuple[float,float,float]],
                                min_tris=20, max_tris=800, top_k=1):
    tri_neighbors, shared_edge_between, edge_to_tris = build_tri_adjacency(triangles)
    # pick worst triangle
    worst_idx = max(range(len(triangle_stress)), key=lambda i: triangle_stress[i])
    region = grow_region(worst_idx, tri_neighbors, min_tris=min_tris, max_tris=max_tris)
    sub_vertices, sub_triangles, sub_tri_to_orig, sub_vert_to_orig = extract_region_submesh(vertices, triangles, region)
    return {
        "seed_tri": worst_idx,
        "region_tris": region,
        "sub_vertices": sub_vertices,
        "sub_triangles": sub_triangles,
        "sub_tri_to_orig": sub_tri_to_orig,
        "sub_vert_to_orig": sub_vert_to_orig,
        "tri_neighbors": tri_neighbors,
        "shared_edge_between": shared_edge_between,
        "edge_to_tris": edge_to_tris,
    }

def compute_triangle_stress_from_atlas(vertices, triangles, atlas_out):
    """
    Compute per-triangle SVD-based stretch (same approach as run_xatlas.py).
    vertices: list of 3D tuples (for the mesh given to atlas_out)
    triangles: list of [i0,i1,i2] into vertices
    atlas_out: (vmapping, out_indices, uvs) returned by run_xatlas functions
    Returns list of ratios (one per triangle) and logs.
    """
    atlas_vmapping, atlas_indices, atlas_uvs = atlas_out
    n = len(triangles)
    ratios = [1.0] * n
    logs = [0.0] * n
    if atlas_vmapping is None or atlas_indices is None or atlas_uvs is None:
        return ratios, logs

    # Build per-original-vertex aggregated UVs
    orig_uvs = defaultdict(list)
    for out_vidx, orig_vidx in enumerate(atlas_vmapping):
        try:
            uv = tuple(map(float, atlas_uvs[out_vidx]))
            orig_uvs[int(orig_vidx)].append(uv)
        except Exception:
            continue

    def avg_uv(uv_list):
        if not uv_list:
            return None
        ux = sum(u for u, v in uv_list) / len(uv_list)
        vx = sum(v for u, v in uv_list) / len(uv_list)
        return (ux, vx)

    EPS_LOCAL = 1e-9
    for i, tri in enumerate(triangles):
        a3 = vertices[tri[0]]; b3 = vertices[tri[1]]; c3 = vertices[tri[2]]
        ua = avg_uv(orig_uvs.get(tri[0], []))
        ub = avg_uv(orig_uvs.get(tri[1], []))
        uc = avg_uv(orig_uvs.get(tri[2], []))
        if ua is None or ub is None or uc is None:
            ratios[i] = 1.0
            logs[i] = 0.0
            continue

        b00 = ub[0] - ua[0]; b10 = ub[1] - ua[1]
        b01 = uc[0] - ua[0]; b11 = uc[1] - ua[1]
        detB = b00 * b11 - b01 * b10
        if abs(detB) <= EPS_LOCAL:
            ratios[i] = 1.0
            logs[i] = 0.0
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

        s_high = _math.sqrt(max(ev_high, EPS_LOCAL))
        s_low  = _math.sqrt(max(ev_low, EPS_LOCAL))

        if s_low <= EPS_LOCAL:
            ratio = 1.0
        else:
            ratio = s_high / s_low
        ratio = min(ratio, 1e6)
        ratios[i] = ratio
        logs[i] = _math.log(ratio + EPS_LOCAL)
    return ratios, logs


def iterative_cut_loop(vertices, triangles, tri_to_face,
                       triangle_stress=None,
                       iterations=4, top_k=1,
                       min_tris=20, max_tris=800,
                       stop_when_below=None,
                       edge_key_to_index=None):
    """
    Minimal iterative loop prototype.
    - vertices, triangles, tri_to_face: full mesh exported structures.
    - triangle_stress: optional initial per-triangle stress list (if None, we compute using run_xatlas on full mesh once).
    - iterations: number of outer iterations.
    - top_k: how many top triangles to attempt per iteration.
    - returns: dict with 'proposed_seam_edges' (set of sorted (v0,v1) tuples) and 'history'.
    Notes: This prototype does NOT mutate mesh or force xatlas to honor seams; it proposes edges to cut.
    """
    tri_neighbors, shared_edge_between, edge_to_tris = build_tri_adjacency(triangles)
    if triangle_stress is None:
        print("iterative_cut_loop: computing initial global stress (one full xatlas run)...")
        tri_chart_ids, atlas_out = run_xatlas.run_xatlas_and_get_chart_per_triangle(vertices, triangles)
        triangle_stress, _ = compute_triangle_stress_from_atlas(vertices, triangles, atlas_out)

    proposed_seams = set()
    resolved_triangles = set()
    history = []

    for it in range(iterations):
        # pick candidate triangles by stress not already resolved
        indexed = [(i, s) for i, s in enumerate(triangle_stress) if i not in resolved_triangles]
        if not indexed:
            break
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [i for i, s in indexed[:top_k]]

        iter_added = set()
        for seed in top_candidates:
            region = grow_region(seed, tri_neighbors, min_tris=min_tris, max_tris=max_tris)
            # extract submesh
            sub_vertices, sub_triangles, sub_tri_to_orig, sub_vert_to_orig = extract_region_submesh(vertices, triangles, region)
            # run xatlas on submesh to get local atlas (no globals changed)
            try:
                sub_tri_chart_ids, sub_atlas_out = run_xatlas.run_xatlas_and_get_chart_per_triangle(sub_vertices, sub_triangles)
            except Exception as e:
                print("iterative_cut_loop: sub xatlas failed:", e)
                resolved_triangles.update(region)
                continue

            # compute local stresses for submesh
            sub_ratios, sub_logs = compute_triangle_stress_from_atlas(sub_vertices, sub_triangles, sub_atlas_out)
            # find worst triangle inside submesh
            worst_local_idx = max(range(len(sub_ratios)), key=lambda i: sub_ratios[i])
            worst_global_tri = sub_tri_to_orig[worst_local_idx]

            print(f"seed {seed}: region_size={len(region)} worst_global_tri={worst_global_tri} local_stress={sub_ratios[worst_local_idx]:.3g}")

            # path to boundary/seam inside original tri graph constrained to region
            path = dijkstra_to_nearest_boundary(worst_global_tri, triangles, vertices,
                                                tri_neighbors, shared_edge_between, edge_to_tris,
                                                seam_edges=proposed_seams, region_mask=region)
            
            cut_edges = []
            if not path:
                btri = worst_global_tri
                tri_verts = triangles[btri]
                # prefer an actual boundary edge
                b_edges = []
                for a,b in ((tri_verts[0],tri_verts[1]), (tri_verts[1],tri_verts[2]), (tri_verts[2],tri_verts[0])):
                    key = tuple(sorted((int(a), int(b))))
                    if len(edge_to_tris.get(key, [])) == 1:
                        b_edges.append(key)
                else:
                    # fallback: shortest edge on triangle
                    best = None; bestlen = None
                    for a,b in ((tri_verts[0],tri_verts[1]), (tri_verts[1],tri_verts[2]), (tri_verts[2],tri_verts[0])):
                        p0 = vertices[a]; p1 = vertices[b]
                        dx=p0[0]-p1[0]; dy=p0[1]-p1[1]; dz=p0[2]-p1[2]
                        l = (dx*dx+dy*dy+dz*dz)**0.5
                        if bestlen is None or l < bestlen:
                            bestlen = l; best = tuple(sorted((int(a),int(b))))
                    if best is not None:
                        cut_edges = [best]
            else:
                cut_edges = triangles_to_edge_path(path, shared_edge_between)

            if not cut_edges:
                # mark region as resolved to avoid repeatedly picking it
                resolved_triangles.update(region)
                continue

            for e in cut_edges:
                proposed_seams.add(e)
                iter_added.add(e)
                # mark adjacent triangles as resolved so we do not pick them repeatedly
                for tri_idx in edge_to_tris.get(e, []):
                    resolved_triangles.add(tri_idx)

        # record iteration summary
        history.append({
            "iteration": it,
            "candidates": top_candidates,
            "added_edges": list(iter_added),
            "num_proposed_seams": len(proposed_seams)
        })
        print(f"iter {it}: candidates {top_candidates} added {len(iter_added)} edges (total proposed {len(proposed_seams)})")

        # stop condition
        if stop_when_below is not None:
            max_remaining = max([triangle_stress[i] for i in range(len(triangle_stress)) if i not in resolved_triangles] or [0.0])
            if max_remaining <= stop_when_below:
                print("iterative_cut_loop: stopping — remaining max stress below threshold")
                break

    # optionally map edges to edge indices if mapping is provided
    edges_as_indices = None
    if edge_key_to_index is not None:
        edges_as_indices = [edge_key_to_index.get(e) for e in proposed_seams if edge_key_to_index.get(e) is not None]

    return {
        "proposed_seam_edges": proposed_seams,
        "proposed_seam_edge_indices": edges_as_indices,
        "history": history
    }

def save_proposed_cuts(export_json_path: str, result: dict, out_name: str = "proposed_cuts.json"):
    """
    Write proposed cuts next to export JSON.
    Format:
      {
        "proposed_edges": [{"verts":[v0,v1], "edge_index": idx_or_null}, ...],
        "history": [...]
      }
    """
    try:
        export = json.loads(open(export_json_path, "r", encoding="utf-8").read())
    except Exception as e:
        raise RuntimeError(f"Failed to load export JSON: {e}")

    # build vertex-pair -> edge index map from export edges
    edge_map = {}
    for e in export.get("edges", []):
        v0 = int(e["verts"][0]); v1 = int(e["verts"][1])
        key = tuple(sorted((v0, v1)))
        edge_map[key] = int(e.get("index", -1))

    items = []
    for key in sorted(result.get("proposed_seam_edges", [])):
        items.append({
            "verts": [int(key[0]), int(key[1])],
            "edge_index": edge_map.get(tuple(sorted(key)))
        })

    out_dir = os.path.dirname(export_json_path)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"proposed_edges": items, "history": result.get("history", [])}, f, indent=2)

    print(f"Wrote proposed cuts to {out_path}")

def save_proposed_as_import_seams(export_json_path: str, result: dict, out_name: str = "import_seams.json"):
    """
    Optional convenience: write edge indices (if available) as a list suitable for your importer.
    If some proposed edges don't map to an original edge index, they are skipped.
    """
    try:
        export = json.loads(open(export_json_path, "r", encoding="utf-8").read())
    except Exception as e:
        raise RuntimeError(f"Failed to load export JSON: {e}")

    edge_map = {}
    for e in export.get("edges", []):
        v0 = int(e["verts"][0]); v1 = int(e["verts"][1])
        key = tuple(sorted((v0, v1)))
        edge_map[key] = int(e.get("index", -1))

    indices = []
    for key in result.get("proposed_seam_edges", []):
        ei = edge_map.get(tuple(sorted(key)))
        if ei is not None and ei >= 0:
            indices.append(ei)

    out_dir = os.path.dirname(export_json_path)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(set(indices))), f, indent=2)

    print(f"Wrote importer-ready seams to {out_path} ({len(indices)} edges)")