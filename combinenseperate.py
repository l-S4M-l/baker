bl_info = {
    "name": "GL Lightmap Generator",
    "author": "S4M",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "category": "Object",
    "location": "View3D > Sidebar > lumaxGL",
    "description": "Generate GL light maps.",
}

import bpy
import numpy as np
import os
import importlib
import sys

if __name__ != "__main__" and __package__:
    importlib.reload(sys.modules[__name__])

VERTEX_GROUP_SPLITTER = "::;::"
DEBUG = True
bake_queue = []
is_baking = False
group_bake_queue = []
group_bake_active = False
group_pipeline_stage = "idle"
group_pipeline_index = 0
group_pipeline_jobs = []

def _process_group_queue():
    global group_bake_active
    global group_pipeline_stage
    global group_pipeline_index
    global group_pipeline_jobs

    if group_bake_active:
        return 0.2

    if len(group_pipeline_jobs) == 0:
        group_pipeline_stage = "idle"
        group_pipeline_index = 0
        return None

    if group_pipeline_stage == "combine":
        if group_pipeline_index >= len(group_pipeline_jobs):
            group_pipeline_stage = "bake"
            group_pipeline_index = 0
            return 0.05

        job_entry = group_pipeline_jobs[group_pipeline_index]
        if job_entry.get("combined_object") is not None:
            group_pipeline_index += 1
            return 0.05

        group_bake_active = True

        def _do_combine():
            global group_bake_active
            try:
                combined_object = combine(job_entry.get("objects"))
                job_entry["combined_object"] = combined_object

                lightmap_uv = ensure_lightmap_uv(combined_object)
                lightmap_uv.active = True
                lightmap_uv.active_render = True

                smart_uv_unwrap(combined_object)
            except Exception as exc:
                print("Group combine failed:", exc)
            group_bake_active = False
            return None

        bpy.app.timers.register(_do_combine, first_interval=0.01)
        return 0.2

    if group_pipeline_stage == "bake":
        if group_pipeline_index >= len(group_pipeline_jobs):
            group_pipeline_stage = "separate"
            group_pipeline_index = 0
            return 0.05

        job_entry = group_pipeline_jobs[group_pipeline_index]
        combined_object = job_entry.get("combined_object")

        if combined_object is None:
            group_pipeline_index += 1
            return 0.05

        if job_entry.get("baked") is True:
            group_pipeline_index += 1
            return 0.05

        if job_entry.get("bake_started") is True:
            return 0.2

        group_bake_active = True

        def group_callback(image, lightmap_group):
            global group_bake_active
            try:
                assign_lightmap_to_materials(combined_object, image)

                if job_entry.get("callback") is not None:
                    job_entry.get("callback")(image, lightmap_group)
            except Exception as exc:
                print("Group bake callback failed:", exc)
            job_entry["baked"] = True
            job_entry["baked_image"] = image
            group_bake_active = False

            if not bpy.app.timers.is_registered(_process_group_queue):
                bpy.app.timers.register(_process_group_queue, first_interval=0.05)

        job_entry["bake_started"] = True
        bake_lightmap_async(
            combined_object,
            job_entry.get("group_item").name,
            resolution=job_entry.get("resolution"),
            callback=group_callback
        )

        return 0.2

    if group_pipeline_stage == "separate":
        if group_pipeline_index >= len(group_pipeline_jobs):
            group_pipeline_stage = "idle"
            group_pipeline_index = 0
            group_pipeline_jobs.clear()
            group_bake_queue.clear()
            return None

        job_entry = group_pipeline_jobs[group_pipeline_index]
        combined_object = job_entry.get("combined_object")

        if job_entry.get("separated") is True:
            group_pipeline_index += 1
            return 0.05

        if combined_object is None:
            job_entry["separated"] = True
            group_pipeline_index += 1
            return 0.05

        group_bake_active = True

        def _do_separate():
            global group_bake_active
            try:
                make_object_active_and_selected(combined_object)
                seperate(combined_object)
            except Exception as exc:
                print("Group separation failed:", exc)
            job_entry["separated"] = True
            group_bake_active = False
            return None

        bpy.app.timers.register(_do_separate, first_interval=0.01)
        return 0.2

    return 0.2

def _process_bake_queue():
    global is_baking

    if is_baking:
        return 0.1  # keep timer alive, wait for bake to finish

    if len(bake_queue) == 0:
        return None  # stop timer, nothing left

    next_item = bake_queue.pop(0)
    obj, lightmap_group, resolution, callback = next_item

    is_baking = True

    def _do_bake():
        global is_baking
        try:
            baker = LightmapBaker(resolution)
            img = baker.run(obj, lightmap_group)
            if callback:
                callback(img, lightmap_group)
        except Exception as e:
            print("Lightmap bake failed:", e)

        is_baking = False
        return None  # finish this bake

    bpy.app.timers.register(_do_bake, first_interval=0.01)
    return 0.2  # check queue again
class LightmapBaker:
    def __init__(self, resolution=2048):
        self.resolution = resolution
        self.obj = None
        self.scene = bpy.context.scene
        self.cycles = None

    def run(self, obj, lightmap_group):
        self.obj = obj
        self._ensure_cycles()
        self.img_combined = self._new_image(f"LM_Combined_{lightmap_group}")
        self.img_albedo = self._new_image(f"LM_Albedo_{lightmap_group}")
        self.img_final = self._new_image(f"LM_Final_{lightmap_group}", non_color=True)
        self._bake_combined()
        self._bake_albedo()
        self._divide()
        return self.img_final

    def _ensure_cycles(self):
        if self.scene.render.engine != "CYCLES":
            raise RuntimeError("Switch to Cycles before baking.")
        self.cycles = self.scene.cycles

    def _new_image(self, name, non_color=False):
        if name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[name])
        img = bpy.data.images.new(name, self.resolution, self.resolution, float_buffer=True)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        return img

    def _materials(self):
        return [s.material for s in self.obj.material_slots if s.material and s.material.use_nodes]

    def _find_output(self, mat):
        for n in mat.node_tree.nodes:
            if isinstance(n, bpy.types.ShaderNodeOutputMaterial):
                return n
        return None

    def _find_albedo_source(self, mat):
        nt = mat.node_tree.nodes
        fallback = (1, 1, 1, 1)
        out = self._find_output(mat)
        bsdf = None
        if out:
            for link in out.inputs["Surface"].links:
                if isinstance(link.from_node, bpy.types.ShaderNodeBsdfPrincipled):
                    bsdf = link.from_node
                    break
        if not bsdf:
            for n in nt:
                if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled):
                    bsdf = n
                    break
        if bsdf:
            base = bsdf.inputs.get("Base Color") or bsdf.inputs.get("Color")
            if base:
                try:
                    fallback = tuple(base.default_value)
                except:
                    pass
                if base.links:
                    node = base.links[0].from_node
                    if isinstance(node, bpy.types.ShaderNodeTexImage):
                        return node, fallback
        for n in nt:
            if isinstance(n, bpy.types.ShaderNodeTexImage):
                return n, fallback
        return None, fallback

    def _apply_targets(self, image):
        created = []
        for mat in self._materials():
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tex.image = image
            mat.node_tree.nodes.active = tex
            created.append((mat, tex))
        return created

    def _remove_targets(self, created):
        for mat, tex in created:
            try:
                if tex.name in mat.node_tree.nodes:
                    mat.node_tree.nodes.remove(tex)
            except:
                pass

    def _bake(self, bake_type, image):
        targets = self._apply_targets(image)
        self.cycles.bake_type = bake_type
        self.cycles.use_bake_clear = True
        self.cycles.use_pass_color = True
        self.cycles.use_pass_direct = True
        self.cycles.use_pass_indirect = True
        with bpy.context.temp_override(
            object=self.obj,
            active_object=self.obj,
            selected_objects=[self.obj],
            selected_editable_objects=[self.obj],
        ):
            bpy.ops.object.bake(type=bake_type)
        self._remove_targets(targets)

    def _bake_combined(self):
        self._bake("COMBINED", self.img_combined)

    def _bake_albedo(self):
        overrides = []
        for mat in self._materials():
            nt = mat.node_tree
            links = nt.links
            out = self._find_output(mat)
            if not out:
                continue
            orig_links = [(l.from_node, l.from_socket) for l in out.inputs["Surface"].links]
            for l in list(out.inputs["Surface"].links):
                links.remove(l)
            tex, fallback = self._find_albedo_source(mat)
            emit = nt.nodes.new("ShaderNodeEmission")
            if tex:
                links.new(tex.outputs["Color"], emit.inputs["Color"])
            else:
                emit.inputs["Color"].default_value = fallback
            links.new(emit.outputs["Emission"], out.inputs["Surface"])
            overrides.append((mat, out, emit, orig_links))
        self._bake("EMIT", self.img_albedo)
        for mat, out, emit, orig_links in overrides:
            nt = mat.node_tree
            links = nt.links
            for l in list(out.inputs["Surface"].links):
                links.remove(l)
            for from_node, from_socket in orig_links:
                links.new(from_socket, out.inputs["Surface"])
            if emit.name in nt.nodes:
                nt.nodes.remove(emit)

    def _divide(self):
        C = np.array(self.img_combined.pixels[:]).reshape(-1, 4)
        A = np.array(self.img_albedo.pixels[:]).reshape(-1, 4)
        A[:, :3] = np.clip(A[:, :3], 1e-4, 1.0)
        L = np.zeros_like(C)
        L[:, :3] = C[:, :3] / A[:, :3]
        L[:, 3] = 1.0
        self.img_final.pixels = L.flatten().tolist()

def bake_lightmap_async(obj, lightmap_group, resolution=2048, callback=None):
    bake_queue.append((obj, lightmap_group, resolution, callback))
    if not bpy.app.timers.is_registered(_process_bake_queue):
        bpy.app.timers.register(_process_bake_queue, first_interval=0.01)

def bake_group_item(group_item, resolution, callback):
    global group_bake_active
    # 1. collect objects
    object_list = []
    index = 0
    group_item:LGX_ListItem = group_item
    for objects in group_item.objects:
        object_list.append(objects.obj)

    if len(object_list) == 0:
        group_bake_active = False

        if not bpy.app.timers.is_registered(_process_group_queue):
            bpy.app.timers.register(_process_group_queue, first_interval=0.05)
        return

    try:
        combined_object = combine(object_list)

        lightmap_uv = ensure_lightmap_uv(combined_object)
        lightmap_uv.active = True
        lightmap_uv.active_render = True

        smart_uv_unwrap(combined_object)

        def group_callback(image, lightmap_group):
            print("ok")

            assign_lightmap_to_materials(combined_object, image)

            make_object_active_and_selected(combined_object)
            seperate(combined_object)

            if callback is not None:
                callback(image, lightmap_group)

            group_bake_active = False

            if not bpy.app.timers.is_registered(_process_group_queue):
                bpy.app.timers.register(_process_group_queue, first_interval=0.05)

        bake_lightmap_async(
            combined_object,
            group_item.name,
            resolution=resolution,
            callback=group_callback
        )
    except Exception as exc:
        group_bake_active = False
        print("Group bake failed:", exc)

        if not bpy.app.timers.is_registered(_process_group_queue):
            bpy.app.timers.register(_process_group_queue, first_interval=0.05)

def queue_group_bakes(group_items, resolution, callback):
    global group_pipeline_stage
    global group_pipeline_index
    global group_pipeline_jobs
    global group_bake_active

    group_pipeline_jobs.clear()
    group_bake_queue.clear()

    index = 0

    while index < len(group_items):
        entry = group_items[index]
        object_list = []
        object_index = 0

        while object_index < len(entry.objects):
            object_list.append(entry.objects[object_index].obj)
            object_index += 1

        if len(object_list) > 0:
            job_entry = {
                "group_item": entry,
                "objects": object_list,
                "resolution": resolution,
                "callback": callback,
                "combined_object": None,
                "bake_started": False,
                "baked": False,
                "separated": False,
            }
            group_pipeline_jobs.append(job_entry)

        index += 1

    group_pipeline_stage = "combine"
    group_pipeline_index = 0
    group_bake_active = False

    if len(group_pipeline_jobs) == 0:
        group_pipeline_stage = "idle"
        return

    if not bpy.app.timers.is_registered(_process_group_queue):
        bpy.app.timers.register(_process_group_queue, first_interval=0.01)

def ensure_lightmap_uv(mesh_object):
    mesh_data = mesh_object.data

    uv_layer = None
    index = 0
    while index < len(mesh_data.uv_layers):
        layer = mesh_data.uv_layers[index]
        if layer.name == "LightMap":
            uv_layer = layer
            break
        index += 1

    if uv_layer is None:
        uv_layer = mesh_data.uv_layers.new(name="LightMap")

    return uv_layer

def prepare_lightmap_uv(mesh_object):
    mesh_data = mesh_object.data

    current_render = mesh_data.uv_layers.active_render
    lightmap_uv = ensure_lightmap_uv(mesh_object)

    mesh_data.uv_layers.active = lightmap_uv
    mesh_data.uv_layers.active_index = mesh_data.uv_layers.find(lightmap_uv.name)

    smart_uv_unwrap(mesh_object)

    if current_render is not None:
        try:
            mesh_data.uv_layers.active_render = current_render
        except Exception:
            pass

    return lightmap_uv

def smart_uv_unwrap(mesh_object):
    ensure_object_mode_is("OBJECT")
    make_object_active_and_selected(mesh_object)

    ensure_object_mode_is("EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.uv.smart_project(
        angle_limit=66,
        island_margin=0.02,
        area_weight=1.0,
        correct_aspect=True,
        scale_to_bounds=True
    )

    ensure_object_mode_is("OBJECT")

def assign_lightmap_to_materials(mesh_object, baked_image):
    slot_index = 0

    while slot_index < len(mesh_object.material_slots):
        slot = mesh_object.material_slots[slot_index]
        material = slot.material

        if material is not None and material.use_nodes:
            node_tree = material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            principled = None
            node_index = 0
            while node_index < len(nodes):
                node = nodes[node_index]
                if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
                    principled = node
                    break
                node_index += 1

            if principled is not None:

                # Find Emission socket (Blender 4 renamed it)
                emission_input = principled.inputs["Emission Color"]

                if emission_input is None:
                    print("No emission input on Principled for material:", material.name)
                    slot_index += 1
                    continue

                # Create texture node
                texture_node = nodes.new("ShaderNodeTexImage")
                texture_node.image = baked_image

                uv:bpy.types.ShaderNodeUVMap = nodes.new("ShaderNodeUVMap")
                uv.uv_map = "LightMap"

                links.new(uv.outputs["UV"], texture_node.inputs["Vector"])

                # Connect texture color → Principled emission input
                links.new(texture_node.outputs["Color"], emission_input)

        slot_index += 1

def ensure_object_mode_is(mode_name):
    current_mode = bpy.context.object.mode
    if current_mode != mode_name:
        bpy.ops.object.mode_set(mode=mode_name)

def make_object_active_and_selected(target_object):
    bpy.ops.object.select_all(action="DESELECT")
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object

def remove_unused_materials(target_object):
    used_material_indices = []
    for polygon in target_object.data.polygons:
        if polygon.material_index not in used_material_indices:
            used_material_indices.append(polygon.material_index)
    total_slots = len(target_object.material_slots)
    for slot_index in range(total_slots - 1, -1, -1):
        if slot_index not in used_material_indices:
            target_object.active_material_index = slot_index
            bpy.ops.object.material_slot_remove()

def combine(selected_objects, ops=None):
    for mesh_object in selected_objects:
        make_object_active_and_selected(mesh_object)
        ensure_object_mode_is('EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        ensure_object_mode_is('OBJECT')
        if len(mesh_object.material_slots) > 0 and mesh_object.material_slots[0].material is not None:
            material_pointer = mesh_object.material_slots[0].material
            material_name = material_pointer.name
        else:
            material_name = "Material"
        vertex_group_name = mesh_object.name + VERTEX_GROUP_SPLITTER + material_name
        vertex_group_reference = mesh_object.vertex_groups.get(vertex_group_name)
        if vertex_group_reference is None:
            vertex_group_reference = mesh_object.vertex_groups.new(name=vertex_group_name)
        all_vertex_indices = []
        for vertex in mesh_object.data.vertices:
            all_vertex_indices.append(vertex.index)
        if len(all_vertex_indices) > 0:
            vertex_group_reference.add(all_vertex_indices, 1.0, 'ADD')
    bpy.ops.object.select_all(action="DESELECT")
    for mesh_object in selected_objects:
        mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = selected_objects[0]
    bpy.ops.object.join()

    active_obj = bpy.context.view_layer.objects.active

    return active_obj

def seperate(active_object: bpy.types.Object, ops=None):
    active_object.name = "SPLITTING OBJECT"

    if active_object is None or active_object.type != "MESH":
        if ops is not None:
            ops.report({'ERROR'}, "Select a mesh that contains vertex groups.")
        return {'CANCELLED'}

    ensure_object_mode_is('OBJECT')

    vertex_groups_to_process = []
    for vertex_group in active_object.vertex_groups:
        vertex_groups_to_process.append(vertex_group)

    for vertex_group in vertex_groups_to_process:

        vertex_group_name = vertex_group.name
        if VERTEX_GROUP_SPLITTER not in vertex_group_name:
            continue

        split_parts = vertex_group_name.split(VERTEX_GROUP_SPLITTER)
        original_object_name = split_parts[0]
        original_material_name = split_parts[1]

        ensure_object_mode_is('EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        ensure_object_mode_is('OBJECT')

        vertex_indices_in_group = []
        for vertex in active_object.data.vertices:
            for vertex_group_link in vertex.groups:
                if vertex_group_link.group == vertex_group.index:
                    vertex_indices_in_group.append(vertex.index)

        for vertex_index in vertex_indices_in_group:
            active_object.data.vertices[vertex_index].select = True

        active_object.data.update()

        ensure_object_mode_is('EDIT')
        bpy.ops.mesh.separate(type='SELECTED')
        ensure_object_mode_is('OBJECT')

        newly_created_objects = []
        for possible_object in bpy.context.selected_objects:
            if possible_object != active_object:
                newly_created_objects.append(possible_object)

        if len(newly_created_objects) == 0:
            print("WARNING: mesh.separate() created no new object for", vertex_group_name)
            continue

        new_separated_object: bpy.types.Object = newly_created_objects[-1]

        # --- FORCE RENAME FIX ---
        existing = bpy.data.objects.get(original_object_name)
        if existing is not None and existing != new_separated_object:
            bpy.data.objects.remove(existing, do_unlink=True)
        # -------------------------

        new_separated_object.name = original_object_name

        new_separated_object.data.materials.clear()
        mat = bpy.data.materials.get(original_material_name)
        new_separated_object.data.materials.append(mat)

        new_separated_object.vertex_groups.clear()

    bpy.data.objects.remove(active_object, do_unlink=True)

    if ops is not None:
        ops.report({'INFO'}, "Meshes successfully restored.")

    return {'FINISHED'}



class LGX_ObjectRef(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(type=bpy.types.Object) # type: ignore

class LGX_ListItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Item") # type: ignore
    objects: bpy.props.CollectionProperty(type=LGX_ObjectRef) # type: ignore
    objects_index: bpy.props.IntProperty(default=0) # type: ignore


class LGX_UL_mylist(bpy.types.UIList):
    bl_idname = "LGX_UL_mylist"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=item.name, icon='GROUP_VERTEX')
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text="")

class LGX_OT_add_item(bpy.types.Operator):
    bl_idname = "lgx.add_item"
    bl_label = "Add"

    def execute(self, context):
        scene = context.scene

        item = scene.my_items.add()
        item.name = "Lightgroup " + str(len(scene.my_items))

        warned_objects = []

        for obj in context.selected_objects:

            already_used = False
            for group in scene.my_items[:-1]:  # skip the new one we just created
                for ref in group.objects:
                    if ref.obj == obj:
                        already_used = True
                        break
                if already_used:
                    break

            if already_used:
                warned_objects.append(obj.name)
                continue

            entry = item.objects.add()
            entry.obj = obj

        if warned_objects:
            self.report({'WARNING'}, "Already in another group: " + ", ".join(warned_objects))

        return {'FINISHED'}

class LGX_OT_remove_item(bpy.types.Operator):
    bl_idname = "lgx.remove_item"
    bl_label = "Remove"

    def execute(self, context):
        scene = context.scene
        idx = scene.my_items_index
        if scene.my_items and 0 <= idx < len(scene.my_items):
            scene.my_items.remove(idx)
            scene.my_items_index = max(0, idx - 1)
        return {'FINISHED'}

class LGX_OT_select_items(bpy.types.Operator):
    bl_idname = "lgx.select_item"
    bl_label = "select items"

    def execute(self, context):
        current_index = context.scene.my_items_index
        for objects in context.scene.my_items[current_index].objects:
            objects.obj.select_set(True)

        return {"FINISHED"}

class LGX_OT_deselect_items(bpy.types.Operator):
    bl_idname = "lgx.deselect_item"
    bl_label = "deselect items"

    def execute(self, context):
        current_index = context.scene.my_items_index
        for objects in context.scene.my_items[current_index].objects:
            obj:bpy.types.Object = objects.obj
            obj.select_set(False)

        return {"FINISHED"}


#combine and restore
class LGX_panel_combine(bpy.types.Operator):
    bl_idname = "lgx.combine"
    bl_label = "Combine Objects (Preserve Types)"

    def execute(self, context):
        selected_mesh_objects = []
        for each_object in context.selected_objects:
            if each_object.type == "MESH":
                selected_mesh_objects.append(each_object)
        combine(selected_mesh_objects, self)
        
        return {"FINISHED"}

class LGX_panel_restore(bpy.types.Operator):
    bl_idname = "lgx.restore"
    bl_label = "Separate Combined Mesh"

    def execute(self, context):
        active_object = context.object
        return seperate(active_object, ops=self)




# baking
class LGX_bake_output_file(bpy.types.PropertyGroup):
    LGX_bake_output_folder: bpy.props.StringProperty(
        name="",
        subtype='DIR_PATH',
        description=""
    ) # type: ignore

class LGX_bake_lightmap(bpy.types.Operator):
    bl_idname = "lgx.bake"
    bl_label = "Bake Lightmap"

    def execute(self, context):
        scene = context.scene
        def save_lightmap(img, lightmap_group):

            base_path = bpy.path.abspath(context.scene.lgx_bake_settings.LGX_bake_output_folder)


            path = os.path.join(base_path, f"lightmap_{lightmap_group}.png")

            img.filepath_raw = path
            img.file_format = 'PNG'
            img.save()
            print("Saved lightmap to:", path)

        queue_group_bakes(
            scene.my_items,
            resolution=2048,
            callback=save_lightmap
        )

        return {'FINISHED'}


#panels
class LGX_PT_panel(bpy.types.Panel):
    bl_label = "LumaxGL"
    bl_idname = "LGX_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LumaxGL"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Objects", icon='MESH_PLANE')

        row = box.row()
        box.template_list(
            "LGX_UL_mylist",
            "",
            scene,
            "my_items",
            scene,
            "my_items_index",
            rows=4,
        )

        row = box.row(align=True)
        row.operator("lgx.add_item", icon='ADD', text="Add")
        row.operator("lgx.remove_item", icon='REMOVE', text="Remove")
        row2 = box.row(align=True)
        row2.operator("lgx.select_item", icon='RESTRICT_SELECT_OFF', text="Select")
        row2.operator("lgx.deselect_item", icon='RESTRICT_SELECT_ON', text="Deselect")
    
        layout.separator()
        layout.separator()
        layout.operator("lgx.bake", text="Bake Lightmap")
        layout.label(text="Bake Output location")
        layout.prop(context.scene.lgx_bake_settings, "LGX_bake_output_folder")

class LGX_PT_debug(bpy.types.Panel):
    bl_label = "Debug"
    bl_idname = "LGX_PT_debug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LumaxGL"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Debug Stuff")

        layout.operator("lgx.combine", text="Combine (Preserve Types)")
        layout.operator("lgx.restore", text="Separate & Restore")
        layout.label(text=f"current selected group {context.scene.my_items_index}")

classes = [
    LGX_ObjectRef,
    LGX_ListItem,
    LGX_bake_output_file,
    LGX_UL_mylist,
    LGX_OT_select_items,
    LGX_OT_deselect_items,

    LGX_panel_combine,
    LGX_panel_restore,
    LGX_OT_add_item,
    LGX_OT_remove_item,
    LGX_bake_lightmap,

    LGX_PT_panel,
]

debug_classes = [
    LGX_PT_debug,
]

if DEBUG == True:
    classes.extend(debug_classes)

def register():
    for class_type in classes:
        bpy.utils.register_class(class_type)

    bpy.types.Scene.lgx_bake_settings = bpy.props.PointerProperty(type=LGX_bake_output_file)
    bpy.types.Scene.my_items = bpy.props.CollectionProperty(type=LGX_ListItem)
    bpy.types.Scene.my_items_index = bpy.props.IntProperty(default=0)


def unregister():
    del bpy.types.Scene.lgx_bake_settings
    del bpy.types.Scene.my_items
    del bpy.types.Scene.my_items_index
    for class_type in reversed(classes):
        bpy.utils.unregister_class(class_type)


if __name__ == "__main__":
    register()
