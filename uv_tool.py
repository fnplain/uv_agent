bl_info = {
    "name": "My UV Tool",
    "author": "(local)",
    "version": (0, 1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > My UV Tool",
    "description": "Small panel with a test unwrap operator",
    "category": "UV",
}

import json
from multiprocessing import context
import urllib.request
import urllib.error
import re

import bpy
import os
import math
from bpy_extras.io_utils import ExportHelper


class MESH_OT_MyCustomUnwrapper(bpy.types.Operator):
    bl_idname = "mesh.my_custom_unwrapper"
    bl_label = "Unwrap My Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def _sanitize_secret(self, text):
        if text is None:
            return ""
        cleaned = str(text).strip().strip('"').strip("'")
        cleaned = cleaned.replace("\r", "").replace("\n", "")
        cleaned = ''.join(ch for ch in cleaned if ch.isprintable())
        return cleaned

    
    def export_mesh_data_for_llm(self, obj, temp_dir):
        import bmesh
        import json

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            bm.free()
            bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'ERROR'}, "No UV layer found!")
            return None

        mesh_info = {
            "object_name": obj.name,
            "vertex_count": len(bm.verts),
            "edge_count": len(bm.edges),
            "face_count": len(bm.faces),
            "vertices": [],
            "edges" : [],
            "faces": []
        }

        
        for v in bm.verts:
            mesh_info["vertices"].append({
                "index": v.index,
                "co": [round(v.co.x, 3), round(v.co.y, 3), round(v.co.z, 3)]
            })

       
        for e in bm.edges:
            # safe: compute angle only when exactly two faces reference the edge
            angle = 0.0
            lf = e.link_faces
            if len(lf) == 2:
                try:
                    angle = math.degrees(e.calc_face_angle())
                except Exception:
                    angle = 0.0
            mesh_info["edges"].append({
                "index": e.index,
                "verts": [v.index for v in e.verts],
                "angle": round(angle, 1),
                "is_sharp": not e.smooth
            })

       
        for f in bm.faces:
            face_data = {
                "index": f.index,
                "verts": [v.index for v in f.verts],
                "uv_coords": []
            }
            for loop in f.loops:
                uv = loop[uv_layer].uv
                face_data["uv_coords"].append([round(uv.x, 3), round(uv.y, 3)])
                
            mesh_info["faces"].append(face_data)

        bmesh.update_edit_mesh(obj.data)
        bm.free()
        bpy.ops.object.mode_set(mode='OBJECT')

        mesh = obj.data
        mesh.calc_loop_triangles()
        triangles = []
        for tri in mesh.loop_triangles:
            triangles.append({
                "verts": [int(v) for v in tri.vertices],
                "orig_face": int(tri.polygon_index)
            })
        mesh_info["triangles"] = triangles

        json_path = os.path.join(temp_dir, "export_data.json")
        with open(json_path, 'w') as f:
            json.dump(mesh_info, f)
            
        print(f"Exported rich data to JSON.")
        return json_path




    def request_ai_seams(self, context, mesh_json_path, prompt, temp_dir):
        scene = context.scene
        endpoint = (scene.myuv_ai_endpoint or "").strip()
        model = (scene.myuv_ai_model or "").strip()
        api_key = self._sanitize_secret(scene.myuv_ai_api_key)
        ai_project = (scene.myuv_ai_project or "").strip()
        ai_org = (scene.myuv_ai_organization or "").strip()
        timeout = max(5, int(scene.myuv_ai_timeout))

        if not endpoint or not model:
            return False, "AI endpoint/model missing."
        requires_key = endpoint.lower().startswith("https://")
        if requires_key and not api_key:
            return False, "API key missing."

        try:
            with open(mesh_json_path, 'r', encoding='utf-8') as mesh_file:
                mesh_payload = mesh_file.read()
        except OSError as exc:
            return False, f"Could not read export_data.json: {exc}"

        request_body = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": "You output only a valid JSON array of integer edge indices for seams.",
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nMesh JSON:\n{mesh_payload}",
                },
            ],
        }

        body_bytes = json.dumps(request_body).encode('utf-8')
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if ai_project:
            headers["OpenAI-Project"] = ai_project
        if ai_org:
            headers["OpenAI-Organization"] = ai_org

        try:
            req = urllib.request.Request(endpoint, data=body_bytes, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode('utf-8', errors='ignore')
            debug_path = os.path.join(temp_dir, "ai_error_response.txt")
            try:
                with open(debug_path, 'w', encoding='utf-8') as debug_file:
                    debug_file.write(error_body)
            except OSError:
                pass
            return False, f"HTTP {exc.code}. Details saved to {debug_path}"
        except Exception as exc:
            return False, f"Request failed: {exc}"

        try:
            parsed = json.loads(response_data)
            raw_content = parsed["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
            return False, "Invalid LLM response format."

        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1]
        if raw_content.endswith("```"):
            raw_content = raw_content.rsplit("\n", 1)[0]

        array_match = re.search(r"\[[\s\d,\-]+\]", raw_content)
        if array_match:
            raw_content = array_match.group(0)

        try:
            seam_values = json.loads(raw_content)
            if not isinstance(seam_values, list):
                return False, "LLM output was not a JSON array."
            seam_indices = []
            for value in seam_values:
                seam_indices.append(int(value))
        except (ValueError, TypeError, json.JSONDecodeError):
            return False, "Could not parse seam index array from LLM output."

        out_path = os.path.join(temp_dir, "import_seams.json")
        with open(out_path, 'w', encoding='utf-8') as seam_file:
            json.dump(sorted(set(seam_indices)), seam_file)

        return True, f"Saved AI seams to {out_path}"









    def capture_model_views(self, obj, out_dir=None):
        print(f"Capturing views for {obj.name}...")
        if out_dir is None:
            blend_dir = bpy.path.abspath("//") or bpy.app.tempdir
            out_dir = os.path.join(blend_dir, "uv_agent_shots")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # set up temp cam
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        bpy.context.scene.camera = cam

        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = max(obj.dimensions) * 1.5

        bpy.context.preferences.view.show_developer_ui = True

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                orig_overlays = space.overlay.show_overlays
                orig_shading = space.shading.type
                
                space.overlay.show_overlays = True
                space.overlay.show_cursor = False
                space.overlay.show_floor = False
                space.overlay.show_axis_x = False
                space.overlay.show_axis_y = False
                space.overlay.show_text = False
                space.overlay.show_wireframes = True


                space.overlay.show_wireframes = True

                space.overlay.show_extra_indices = True 
                
                space.region_3d.view_perspective = 'CAMERA'

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_all(action='SELECT')

        views = {
            "front":  (math.radians(90), 0, 0),
            "back":   (math.radians(90), 0, math.radians(180)),
            "left":   (math.radians(90), 0, math.radians(-90)),
            "right":  (math.radians(90), 0, math.radians(90)),
            "top":    (0, 0, 0),
            "bottom": (math.radians(180), 0, 0)
        }


        dist = max(obj.dimensions) * 2

        for name, rot in views.items():
            cam.rotation_euler = rot
            if name == "front":  cam.location = (0, -dist, 0)
            elif name == "back": cam.location = (0, dist, 0)
            elif name == "top":   cam.location = (0, 0, dist)
            elif name == "bottom": cam.location = (0, 0, -dist)
            elif name == "left":  cam.location = (-dist, 0, 0)
            elif name == "right": cam.location = (dist, 0, 0)

            bpy.context.view_layer.update()

            file_path = os.path.join(out_dir, f"view_{name}.png")
            bpy.context.scene.render.filepath = file_path

            bpy.ops.render.opengl(write_still=True)
            print(f"Saved view {name} to {file_path}")

        #restore view settings

        bpy.ops.object.mode_set(mode='OBJECT')

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.region_3d.view_perspective = 'PERSP'
                space.overlay.show_extra_indices = False

                space.overlay.show_overlays = orig_overlays
                space.shading.type = orig_shading
        
        bpy.data.objects.remove(cam, do_unlink=True)
        return out_dir








    def execute(self, context):
        print("Button was clicked!")

        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected!")
            return {'CANCELLED'}

        blend_dir = bpy.path.abspath("//") or bpy.app.tempdir
        temp_dir = os.path.join(blend_dir, "uv_agent_shots")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        print("UV Agent export folder:", temp_dir)

        if not context.scene.myuv_auto_call_ai:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01)
            bpy.ops.object.mode_set(mode='OBJECT')

        uv_path = os.path.join(temp_dir, "uv_layout.png")

        image_area = None
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                image_area = area
                break

        if image_area is None:
            image_area = context.area
            orig_type = image_area.type
            image_area.type = 'IMAGE_EDITOR'

        try:
            # Turn on stretch in the UI for the user to see
            space = image_area.spaces.active
            if hasattr(space, "show_stretch"):
                space.show_stretch = True
            with context.temp_override(area=image_area):
                bpy.ops.uv.export_layout(filepath=uv_path, size=(2048, 2048), opacity=1.0)
        finally:
            if image_area is not None and image_area.type == 'IMAGE_EDITOR' and 'orig_type' in locals():
                image_area.type = orig_type

        # Ensure the UV editor shows stretch if it exists
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                space = area.spaces.active
                if hasattr(space, "show_stretch"):
                    space.show_stretch = True
                break


        # old method, which just set all edges as seams, which is not ideal

        # import bmesh
        # bm = bmesh.from_edit_mesh(obj.data)
        # angle_threshold = math.radians(45) 

        # for edge in bm.edges:
        #     if len(edge.link_faces) == 2:
        #         angle = edge.calc_face_angle()
        #         if angle > angle_threshold:
        #             edge.seam = True
        #         else:
        #             edge.seam = False

        # bmesh.update_edit_mesh(obj.data)
        # bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.01)
        # old method end 

        # self.report({'INFO'}, "Smart UVs and Capturing views...")
        # self.capture_model_views(obj, temp_dir)
        mesh_json_path = self.export_mesh_data_for_llm(obj, temp_dir)
        if mesh_json_path is None:
            return {'CANCELLED'}
        
        # Output standard prompt
        prompt = """Analyze the attached 3D mesh data (export_data.json) and orthogonal rendered views. 
Determine the optimal UV seam layout to minimize UV stretching and hide seams in less visible areas.
Take into account sharp edges and geometry flow.

Output exactly ONE thing: a raw JSON array format containing the integer edge indices that should be marked as seams.
Do NOT output code blocks (like ```json). DO NOT output any conversational text.
Example format:
[0, 15, 23, 42]"""
        
        prompt_path = os.path.join(temp_dir, "prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write(prompt)

        if context.scene.myuv_auto_call_ai:
            ok, message = self.request_ai_seams(context, mesh_json_path, prompt, temp_dir)
            if ok:
                self.report({'INFO'}, message)
            else:
                self.report({'WARNING'}, f"AI call failed: {message}. Manual files are still ready.")

        self.report({'INFO'}, f"Draft ready. Files saved to {temp_dir}")
        return {'FINISHED'}


class MESH_OT_ApplyProposedCuts(bpy.types.Operator):
    bl_idname = "mesh.apply_proposed_cuts"
    bl_label = "Apply Proposed Cuts"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import bmesh
        import json

        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected!")
            return {'CANCELLED'}

        blend_dir = bpy.path.abspath("//") or bpy.app.tempdir
        temp_dir = os.path.join(blend_dir, "uv_agent_shots")
        proposed_path = os.path.join(temp_dir, "proposed_cuts.json")
        import_path = os.path.join(temp_dir, "import_seams.json")

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        # Prefer import_seams.json (edge indices), fallback to proposed_cuts.json (vertex pairs)
        if os.path.exists(import_path):
            try:
                with open(import_path, 'r', encoding='utf-8') as f:
                    indices = json.load(f)
                seam_indices = set(int(i) for i in indices)
            except Exception as exc:
                self.report({'ERROR'}, f"Failed to read import_seams.json: {exc}")
                bpy.ops.object.mode_set(mode='OBJECT')
                return {'CANCELLED'}

            applied = 0
            for edge in bm.edges:
                if edge.index in seam_indices and not edge.seam:
                    edge.seam = True
                    applied += 1

            bmesh.update_edit_mesh(obj.data)
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.01)
            bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'INFO'}, f"Applied {applied} new seams from import_seams.json")
            return {'FINISHED'}

        if os.path.exists(proposed_path):
            try:
                with open(proposed_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                pairs = [tuple(sorted(item.get('verts', []))) for item in data.get('proposed_edges', [])]
            except Exception as exc:
                self.report({'ERROR'}, f"Failed to read proposed_cuts.json: {exc}")
                bpy.ops.object.mode_set(mode='OBJECT')
                return {'CANCELLED'}

            # map vertex-pair -> bmesh edge
            vertpair_to_edge = {}
            for e in bm.edges:
                a = e.verts[0].index; b = e.verts[1].index
                vertpair_to_edge[tuple(sorted((a, b)))] = e

            applied = 0
            for key in pairs:
                e = vertpair_to_edge.get(tuple(key))
                if e and not e.seam:
                    e.seam = True
                    applied += 1

            bmesh.update_edit_mesh(obj.data)
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.01)
            bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'INFO'}, f"Applied {applied} proposed seams from proposed_cuts.json")
            return {'FINISHED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'ERROR'}, "No import_seams.json or proposed_cuts.json found in uv_agent_shots")
        return {'CANCELLED'}

class MESH_OT_ImportSeams(bpy.types.Operator):
    bl_idname = "mesh.import_seams"
    bl_label = "Apply AI Seams"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import bmesh
        import json

        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected!")
            return {'CANCELLED'}

        blend_dir = bpy.path.abspath("//") or bpy.app.tempdir
        temp_dir = os.path.join(blend_dir, "uv_agent_shots")
        json_path = os.path.join(temp_dir, "import_seams.json")

        if not os.path.exists(json_path):
            self.report({'ERROR'}, f"Seams file not found: {json_path}")
            return {'CANCELLED'}

        with open(json_path, 'r') as f:
            content = f.read().strip()

        # Clean up potential LLM markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("\n", 1)[0]

        try:
            # Expecting a flat list like [0, 4, 12, 15]
            seam_indices = set(json.loads(content))
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Invalid JSON in seams file.")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        # Apply seams
        for edge in bm.edges:
            edge.seam = edge.index in seam_indices

        bmesh.update_edit_mesh(obj.data)

        # Unwrap with the new seams
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.01)
        bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, f"Applied {len(seam_indices)} seams and unwrapped!")
        return {'FINISHED'}

class VIEW3D_PT_MyUVPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'My UV Tool'
    bl_label = 'UV Controls'

    def draw(self, context):
        layout = self.layout
        layout.operator(MESH_OT_MyCustomUnwrapper.bl_idname, text="Export Mesh (no AI)")
        layout.operator(MESH_OT_ApplyProposedCuts.bl_idname, text="Apply Proposed Cuts")
        layout.operator(MESH_OT_ImportSeams.bl_idname, text="Apply Import Seams (raw)")
        layout.separator()
        layout.operator(MESH_OT_ExportSeamGPTData.bl_idname, text="Export SeamGPT Data")
        # layout.prop(context.scene, "myuv_auto_call_ai")
        # layout.prop(context.scene, "myuv_ai_endpoint")
        # layout.prop(context.scene, "myuv_ai_model")
        # layout.prop(context.scene, "myuv_ai_api_key")
        # layout.prop(context.scene, "myuv_ai_project")
        # layout.prop(context.scene, "myuv_ai_organization")
        # layout.prop(context.scene, "myuv_ai_timeout")


class MESH_OT_ExportSeamGPTData(bpy.types.Operator, ExportHelper):
    bl_idname = "mesh.export_seamgpt"
    bl_label = "Export SeamGPT Data"
    bl_description = "Exports active object seam data to SeamGPT JSON format"
    filename_ext = ".json"

    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        import json
        import mathutils

        TARGET = 15360


        obj = context.active_object
        if obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh.")
            return {'CANCELLED'}
        mesh = obj.data
        

        mesh_metadata = {
            "name": obj.name,
            "poly_count": len(mesh.polygons)
        }


        verts_world = []
        for v in mesh.vertices:
            wc = obj.matrix_world @ v.co
            verts_world.append([float(wc.x), float(wc.y), float(wc.z)])


        edge_midpoints = []
        for e in mesh.edges:
            a = obj.matrix_world @ mesh.vertices[e.vertices[0]].co
            b = obj.matrix_world @ mesh.vertices[e.vertices[1]].co
            mid = (a + b) / 2
            edge_midpoints.append([float(mid.x), float(mid.y), float(mid.z)])



        def sample_repeat(points, target):
            if not points:
                return [[0.0, 0.0, 0.0]] * target
            n = len(points)
            if n >= target:
                step = n / target
                return [[round(c, 6) for c in points[int(i * step)]] for i in range(target)]
            res = []
            i = 0
            while len(res) < target:
                p = points[i % n]
                res.append([round(p[0], 6), round(p[1], 6), round(p[2], 6)])
                i += 1
            return res
        
        vertex_points = sample_repeat(verts_world, TARGET)
        edge_points = sample_repeat(edge_midpoints, TARGET)
        
        seam_edges = []
        ordered_list = []
        
        for e in mesh.edges:
            if e.use_seam:
                a = int(e.vertices[0])
                b = int(e.vertices[1])
                seam_edges.append((a, b))
                
                v1 = obj.matrix_world @ mesh.vertices[a].co
                v2 = obj.matrix_world @ mesh.vertices[b].co
                
                start_coord, end_coord = sorted([list(v1), list(v2)], key=lambda x: (x[1], x[2], x[0]))
                
                if (Vector(start_coord) - v1).length < (Vector(start_coord) - v2).length:
                    start_idx, end_idx = a, b
                else:
                    start_idx, end_idx = b, a    
                ordered_list.append(((start_coord[1], start_coord[2], start_coord[0]), [start_idx, end_idx]))
                
        ordered_list.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
        ordered_segments = [pair for _, pair in ordered_list]
        
        
        data = {
            "mesh_metadata": mesh_metadata,
            "shape_context": {
                "vertex_points": vertex_points,
                "edge_points": edge_points
            },
            "labels": { 
                "seam_edges": seam_edges,
                "ordered_segments": ordered_segments
            }
        }
        
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to write file: {exc}")
            return {'CANCELLED'}
         
        self.report({'INFO'}, f"Saved SeamGPT data to {self.filepath}")
        return {'FINISHED'} 

classes = (
    MESH_OT_MyCustomUnwrapper,
    MESH_OT_ImportSeams,
    MESH_OT_ApplyProposedCuts,
    MESH_OT_ExportSeamGPTData,
    VIEW3D_PT_MyUVPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.myuv_auto_call_ai = bpy.props.BoolProperty(
        name="Auto call AI",
        description="Call LLM API after export to generate import_seams.json automatically",
        default=False,
    )
    bpy.types.Scene.myuv_ai_endpoint = bpy.props.StringProperty(
        name="AI Endpoint",
        description="OpenAI-compatible chat completions endpoint",
        default="https://api.openai.com/v1/chat/completions",
    )
    bpy.types.Scene.myuv_ai_model = bpy.props.StringProperty(
        name="Model",
        description="Model name for your provider",
        default="gpt-4o-mini",
    )
    bpy.types.Scene.myuv_ai_api_key = bpy.props.StringProperty(
        name="API Key",
        description="Bearer API key for your provider",
        subtype='PASSWORD',
        default="",
    )
    bpy.types.Scene.myuv_ai_project = bpy.props.StringProperty(
        name="OpenAI Project (optional)",
        description="Optional OpenAI project id (for project-scoped keys)",
        default="",
    )
    bpy.types.Scene.myuv_ai_organization = bpy.props.StringProperty(
        name="OpenAI Org (optional)",
        description="Optional OpenAI organization id",
        default="",
    )
    bpy.types.Scene.myuv_ai_timeout = bpy.props.IntProperty(
        name="Timeout (sec)",
        description="LLM request timeout in seconds",
        default=60,
        min=5,
        max=300,
    )
    print("Addon registered successfully!")


def unregister():
    del bpy.types.Scene.myuv_ai_timeout
    del bpy.types.Scene.myuv_ai_organization
    del bpy.types.Scene.myuv_ai_project
    del bpy.types.Scene.myuv_ai_api_key
    del bpy.types.Scene.myuv_ai_model
    del bpy.types.Scene.myuv_ai_endpoint
    del bpy.types.Scene.myuv_auto_call_ai
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Addon unregistered successfully!")


if __name__ == "__main__":
    register()