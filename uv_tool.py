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

import bpy
import os
import math


class MESH_OT_MyCustomUnwrapper(bpy.types.Operator):
    bl_idname = "mesh.my_custom_unwrapper"
    bl_label = "Unwrap My Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    
    def export_mesh_data_for_llm(self, obj, temp_dir):
        import bmesh
        import json

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            self.report({'ERROR'}, "No UV layer found!")
            return

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
            angle = 0
            if not e.is_boundary:
                angle = math.degrees(e.calc_face_angle())
                
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

        json_path = os.path.join(temp_dir, "export_data.json")
        with open(json_path, 'w') as f:
            json.dump(mesh_info, f)
            
        print(f"Exported rich data to JSON.")
        return json_path









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

        self.report({'INFO'}, "Smart UVs and Capturing views...")
        self.capture_model_views(obj, temp_dir)
        self.export_mesh_data_for_llm(obj, temp_dir)
        
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

        self.report({'INFO'}, f"Draft ready. Files saved to {temp_dir}")
        return {'FINISHED'}

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
        layout.operator(MESH_OT_MyCustomUnwrapper.bl_idname)
        layout.operator(MESH_OT_ImportSeams.bl_idname)

classes = (
    MESH_OT_MyCustomUnwrapper,
    MESH_OT_ImportSeams,
    VIEW3D_PT_MyUVPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    print("Addon registered successfully!")


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Addon unregistered successfully!")


if __name__ == "__main__":
    register()