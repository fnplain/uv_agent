# test_xatlas.py
import sys
print("python:", sys.executable)
try:
    import xatlas
    print("import xatlas ok:", getattr(xatlas, '__version__', 'no-version-attr'))
except Exception as e:
    print("import xatlas failed:", e); raise

# small mesh
verts = [[0,0,0],[1,0,0],[0,1,0]]
tris = [(0,1,2)]

# try parametrize
try:
    vm, out, uvs = xatlas.parametrize(verts, tris)
    print("parametrize -> vmapping:", len(vm), "out_indices:", len(out), "uvs:", len(uvs))
except Exception as e:
    print("parametrize failed:", e)

# try Atlas API
try:
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, tris)
    atlas.generate()
    try:
        vm2, indices2, uvs2 = atlas[0]
        print("atlas[0] -> vmapping:", len(vm2), "indices:", len(indices2), "uvs:", len(uvs2))
    except Exception as e:
        print("atlas[0] not available or failed:", e)
    try:
        ai, ci = atlas.get_mesh_vertex_assignement(0)
        print("get_mesh_vertex_assignement -> lengths:", len(ai), len(ci))
    except Exception as e:
        print("get_mesh_vertex_assignement failed:", e)
except Exception as e:
    print("Atlas() path failed:", e)