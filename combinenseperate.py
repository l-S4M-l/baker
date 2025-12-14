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
import math
from math import inf
from scipy.ndimage import gaussian_filter, binary_erosion
from PIL import Image, ImageMath

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
group_pipeline_complete_callback = None

global baked_object
baked_object:list[bpy.types.Object] = []


###########
#   BAKE
###########
def _process_group_queue():
    global group_bake_active
    global group_pipeline_stage
    global group_pipeline_index
    global group_pipeline_jobs
    global group_pipeline_complete_callback

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
                lightmap_uv.active_render = False

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
        # When all jobs have been separated
        if group_pipeline_index >= len(group_pipeline_jobs):
            group_pipeline_stage = "idle"
            group_pipeline_index = 0

            # Copy results before clearing
            finished_jobs = list(group_pipeline_jobs)

            group_pipeline_jobs.clear()
            group_bake_queue.clear()

            # Fire completion callback if set
            if group_pipeline_complete_callback is not None:
                try:
                    group_pipeline_complete_callback(finished_jobs)
                except Exception as exc:
                    print("Group pipeline completion callback failed:", exc)

            group_pipeline_complete_callback = None
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
                old_objects = seperate(combined_object)
                # store separated objects on the job entry
                job_entry["separated_objects"] = old_objects
            except Exception as exc:
                print("Group separation failed:", exc)
                job_entry["separated_objects"] = None
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
            raise(e)

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
        self.img_mask = self._new_image(f"LM_mask_{lightmap_group}")
        self._bake_combined()
        self._bake_albedo()
        self._bake_mask()
        max_range = bpy.context.scene.lgx_render_light_level.value
        self._divide(max_range=max_range)
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
    
    def _enable_best_gpu(self):
        prefs = bpy.context.preferences.addons['cycles'].preferences

        # Try OPTIX first
        if 'OPTIX' in prefs.get_device_types():
            prefs.compute_device_type = 'OPTIX'
        # Fall back to CUDA
        elif 'CUDA' in prefs.get_device_types():
            prefs.compute_device_type = 'CUDA'
        # Last resort: CPU
        else:
            prefs.compute_device_type = 'NONE'

        # Enable all devices of the selected type
        for device in prefs.get_devices()[0]:
            device.use = True

    def _bake(self, bake_type, image, samples=10):
    # Set bake samples
        #self._enable_best_gpu()

        bpy.context.scene.cycles.samples = samples

        bpy.context.scene.render.bake.margin = 0  # pixels of bleed
        bpy.context.scene.render.bake.margin_type = 'ADJACENT_FACES'  # best quality

        targets = self._apply_targets(image)
        self.cycles.bake_type = bake_type
        self.cycles.use_bake_clear = True
        self.cycles.use_pass_color = True
        self.cycles.use_pass_direct = False
        self.cycles.use_pass_indirect = True
        self.cycles.use_denoising = True
        self.cycles.filter_width = 1.0

        with bpy.context.temp_override(
            object=self.obj,
            active_object=self.obj,
            selected_objects=[self.obj],
            selected_editable_objects=[self.obj],
        ):
            bpy.ops.object.bake(type=bake_type)
        
        self._remove_targets(targets)

    def _bake_combined(self):
        samples = bpy.context.scene.lgx_bake_samples.value
        self._bake("COMBINED", self.img_combined, samples=samples)

        albedo = np.array(self.img_combined.pixels[:], dtype=np.float32).reshape(512, 512, 4)
        _debug_save_image(albedo, "COMBINED")

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

    def _bake_mask(self):
        """
        Bake a mask:
        - If BSDF Alpha has a texture → use that alpha directly as mask
        - If no alpha → pure white mask
        """

        overrides = []

        for mat in self._materials():
            nt = mat.node_tree
            links = nt.links

            out = self._find_output(mat)
            if not out:
                continue

            # Save original surface links
            orig_links = [(l.from_node, l.from_socket)
                        for l in out.inputs["Surface"].links]

            # Remove them
            for l in list(out.inputs["Surface"].links):
                links.remove(l)

            # Find BSDF + alpha
            bsdf = None
            for n in nt.nodes:
                if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled):
                    bsdf = n
                    break

            alpha_source = None
            if bsdf:
                alpha_input = bsdf.inputs.get("Alpha")
                if alpha_input and alpha_input.is_linked:
                    alpha_source = alpha_input.links[0].from_socket

            # ----------------------------------------------------
            # Build Emission for mask bake
            # ----------------------------------------------------
            emit = nt.nodes.new("ShaderNodeEmission")

            if alpha_source:
                # Use alpha directly as the emission color
                # Alpha socket → Emission.Color
                links.new(alpha_source, emit.inputs["Color"])

            else:
                # Pure white mask fallback
                emit.inputs["Color"].default_value = (1, 1, 1, 1)

            emit.inputs["Strength"].default_value = 1.0

            # Connect emission → Surface
            links.new(emit.outputs["Emission"], out.inputs["Surface"])

            overrides.append((mat, out, emit, orig_links))

        # Bake EMIT → img_mask
        self._bake("EMIT", self.img_mask)

        # Cleanup + restore
        for mat, out, emit, orig_links in overrides:
            nt = mat.node_tree
            links = nt.links

            # Remove emission link
            for l in list(out.inputs["Surface"].links):
                links.remove(l)

            for from_node, from_socket in orig_links:
                links.new(from_socket, out.inputs["Surface"])

            # Remove temporary emission node
            if emit.name in nt.nodes:
                nt.nodes.remove(emit)

        print("✔ Mask bake complete (direct alpha mask).")

    def _divide(self, max_range=3.5):
        """
        Perform Photoshop-style Divide (Combined / Albedo) using shader nodes
        directly on the baked object, using the 'LightMap' UV set.
        """

        obj = self.obj
        nt_created = []   # store (mat, node) for cleanup
        link_backup = []  # (mat, out, orig_links)

        # ------------------------------------------------------------
        # Sanity: check LightMap UV exists on this mesh
        # ------------------------------------------------------------
        uv_layer_names = {uv.name for uv in obj.data.uv_layers}
        use_lightmap_uv = "LightMap" in uv_layer_names
        if not use_lightmap_uv:
            print("⚠ No 'LightMap' UV found on", obj.name, "- Divide will use default UVs")

        bake_img = self.img_final

        # ============================================================
        # 1. Override materials with Divide → Emission
        # ============================================================
        for mat in self._materials():
            nt = mat.node_tree
            nodes = nt.nodes
            links = nt.links

            # Find material output
            out = self._find_output(mat)
            if not out:
                continue

            # Backup original Surface links
            orig_links = [(l.from_node, l.from_socket) for l in out.inputs["Surface"].links]
            link_backup.append((mat, out, orig_links))

            # Remove original Surface links
            for l in list(out.inputs["Surface"].links):
                links.remove(l)

            # ---------------- UV node ----------------
            uv_node = None
            if use_lightmap_uv:
                uv_node = nodes.new("ShaderNodeUVMap")
                uv_node.uv_map = "LightMap"
                uv_node.label = "LM UV"
                uv_node.location = (-800, 40)
                nt_created.append((mat, uv_node))

            # ---------------- Textures ----------------
            tex_C = nodes.new("ShaderNodeTexImage")  # Combined
            tex_C.image = self.img_combined
            tex_C.label = "LM Combined"
            tex_C.interpolation = 'Linear'
            tex_C.location = (-600, 140)
            nt_created.append((mat, tex_C))

            tex_A = nodes.new("ShaderNodeTexImage")  # Albedo
            tex_A.image = self.img_albedo
            tex_A.label = "LM Albedo"
            tex_A.interpolation = 'Linear'
            tex_A.location = (-600, -40)
            nt_created.append((mat, tex_A))

            # Feed LightMap UV into both textures
            if uv_node is not None:
                links.new(uv_node.outputs["UV"], tex_C.inputs["Vector"])
                links.new(uv_node.outputs["UV"], tex_A.inputs["Vector"])

            # ---------------- Divide mix ----------------
            mix = nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "DIVIDE"
            mix.inputs["Fac"].default_value = 1.0
            mix.location = (-300, 40)
            mix.label = "Combined / Albedo"
            nt_created.append((mat, mix))

            # Combined → Color1, Albedo → Color2
            links.new(tex_C.outputs["Color"], mix.inputs["Color1"])
            links.new(tex_A.outputs["Color"], mix.inputs["Color2"])

            # ---------------- Map Range (HDR → 0–1, VECTOR) ----------------
            map_range = nodes.new("ShaderNodeMapRange")
            map_range.label = f"MapRange 0-{max_range} → 0-1"
            map_range.location = (-120, 40)

            # IMPORTANT: operate on RGB vector, not scalar
            map_range.data_type = 'FLOAT_VECTOR'
            map_range.clamp = False

            # From / To ranges
            map_range.inputs["From Min"].default_value = (0.0, 0.0, 0.0)
            map_range.inputs["From Max"].default_value = (max_range, max_range, max_range)
            map_range.inputs["To Min"].default_value = (0.0, 0.0, 0.0)
            map_range.inputs["To Max"].default_value = (1.0, 1.0, 1.0)

            nt_created.append((mat, map_range))

            links.new(mix.outputs["Color"], map_range.inputs["Vector"])


            # ---------------- Emission + Output ----------------
            emit = nodes.new("ShaderNodeEmission")
            emit.location = (60, 40)
            emit.label = "Divide Emit"
            nt_created.append((mat, emit))

            links.new(map_range.outputs["Vector"], emit.inputs["Color"])
            links.new(emit.outputs["Emission"], out.inputs["Surface"])

            # ---------------- Bake Target node ----------------
            tex_bake = nodes.new("ShaderNodeTexImage")
            tex_bake.image = bake_img
            tex_bake.label = "Bake Target"
            tex_bake.location = (150, -120)
            nt_created.append((mat, tex_bake))

            nt.nodes.active = tex_bake

        # ============================================================
        # 2. Bake EMIT (Divide result) into self.img_final
        # ============================================================
        with bpy.context.temp_override(
            object=obj,
            active_object=obj,
            selected_objects=[obj],
            selected_editable_objects=[obj],
        ):
            bpy.ops.object.bake(
                type='EMIT',
                margin=4,
                use_clear=True
            )

        # ============================================================
        # 3. Restore original materials & clean up nodes
        # ============================================================
        for mat, out, orig_links in link_backup:
            nt = mat.node_tree
            links = nt.links

            # Remove our temporary EMIT links
            for l in list(out.inputs["Surface"].links):
                links.remove(l)

            # Restore original links
            for from_node, from_socket in orig_links:
                links.new(from_socket, out.inputs["Surface"])

        # Delete all temp nodes we created
        for mat, node in nt_created:
            try:
                mat.node_tree.nodes.remove(node)
            except:
                pass

        print("✔ Divide bake complete (LightMap UV linked)")
        composite_blurred_mask(self.img_final, self.img_mask)


def _debug_save_image(arr, stepname): #turned off rn
    return
    """Save a numpy RGBA array as PNG for debugging."""
    h, w, c = arr.shape

    # Create temp image
    img = bpy.data.images.new(f"_debug_{stepname}", width=w, height=h, alpha=True, float_buffer=True)

    # Flatten and assign
    flat = np.clip(arr.reshape(-1), 0.0, 1.0).tolist()
    img.pixels[:] = flat

    # Build path
    base_path = bpy.path.abspath(bpy.context.scene.lgx_bake_settings.LGX_bake_output_folder)
    path = os.path.join(base_path, f"lightmap_{stepname}.png")

    # Save it
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()

    print(f"✔ Saved debug step: {path}")

    # Cleanup
    bpy.data.images.remove(img, do_unlink=True)

#doing levels 
def apply_levels(img, in_black=0.0, in_white=1.0, gamma=1.0,
                 out_black=0.0, out_white=1.0):
    """
    Photoshop Levels for both 2D grayscale (H,W)
    and 3D RGB(A) arrays (H,W,C).
    """

    img = img.astype(np.float32)

    # Normalize input range
    norm = (img - in_black) / (in_white - in_black)
    norm = np.clip(norm, 0.0, 1.0)

    # Gamma correction
    if gamma != 1.0:
        norm = np.power(norm, gamma)

    # Output range
    out = out_black + norm * (out_white - out_black)

    return np.clip(out, 0.0, 1.0)

def composite_blurred_mask(final_img, mask_img,
                           feather_px=1, contract_px=3, blur_px=3):
    """
    Photoshop-style compositing pipeline:

    1. Contract mask (Shift Edge)
    2. Feather mask (Gaussian blur)
    3. Transparent-aware RGBA blur (Photoshop behavior)
    4. Sharp-over-blurred composite using the processed mask
    5. Write result back to Blender image

    Includes: RGBA blur, per-channel normalization, and full debug saving.
    """

    # ======================================================
    # LOAD IMAGES
    # ======================================================
    w, h = final_img.size

    final = np.array(final_img.pixels[:], dtype=np.float32).reshape(h, w, 4)


    mask = np.array(mask_img.pixels[:], dtype=np.float32).reshape(h, w, 4)

    mask = mask[:, :, 0]  # 1-channel (H,W)

    crisp_mask = mask

    # ======================================================
    # 1. CONTRACT MASK
    # ======================================================

    _debug_save_image(np.stack([mask, mask, mask, np.ones_like(mask)], axis=2), "crisp mask")
    if contract_px > 0:
        binmask = mask > 0.5
        for _ in range(contract_px):
            binmask = binary_erosion(binmask)
        mask = binmask.astype(np.float32)

    _debug_save_image(np.stack([mask, mask, mask, np.ones_like(mask)], axis=2), "contracted")

    # ======================================================
    # 2. FEATHER MASK
    # ======================================================
    if feather_px > 0:
        mask = gaussian_filter(mask, sigma=feather_px)

    mask = np.clip(mask, 0, 1)

    # ======================================================
    # 3. TRANSPARENT-AWARE RGBA BLUR
    # ======================================================
    h, w, _ = final.shape
    mask2d = crisp_mask  # (H,W)

    blurred_channels = []

    # ---- RGB channels ----
    for c in range(3):
        channel = final[:, :, c]

        # numerator = gaussian( channel * mask )
        num = gaussian_filter(channel * mask2d, sigma=blur_px)

        # denominator = gaussian(mask)
        den = gaussian_filter(mask2d, sigma=blur_px)
        den = np.clip(den, 1e-6, 1.0)

        blurred_channels.append(num / den)

    # ---- Alpha channel (blur independently) ----
    alpha_channel = final[:, :, 3]
    num_a = gaussian_filter(alpha_channel * mask2d, sigma=blur_px)
    den_a = gaussian_filter(mask2d, sigma=blur_px)
    den_a = np.clip(den_a, 1e-6, 1.0)
    blurred_alpha = num_a / den_a

    # Combined blurred RGBA
    blurred_rgba = np.stack(blurred_channels + [blurred_alpha], axis=2)

    _debug_save_image(blurred_rgba, "blurred_underlayer_rgba")



    # ======================================================
    # 4. SHARP TOP LAYER (unpremultiply in NumPy)
    # ======================================================
    sharp = final.copy()    # float32 (H,W,4)
    alpha = sharp[:, :, 3:4]   # shape (H,W,1)

    # unpremultiply
    sharp_rgb = sharp[:, :, :3] / np.clip(alpha, 1e-6, 1.0)
    sharp_rgb = np.clip(sharp_rgb, 0, 1)

    # ======================================================
    # 5. Convert blurred and sharp to uint8 for Pillow
    # ======================================================
    sharp_pil = Image.fromarray(
        (np.dstack([sharp_rgb, alpha]) * 255).astype(np.uint8),
        "RGBA"
    )

    blur_pil = Image.fromarray(
        (np.clip(blurred_rgba, 0, 1) * 255).astype(np.uint8),
        "RGBA"
    )

    # Mask must be grayscale 8-bit
    mask_pil = Image.fromarray(
        (np.clip(mask, 0, 1) * 255).astype(np.uint8),
        "L"
    )

    # ======================================================
    # 6. Photoshop-style composite:  Sharp over Blurred using Mask
    # ======================================================
    result_pil = Image.composite(sharp_pil, blur_pil, mask_pil)

    # Convert result back to NumPy float32
    out = np.array(result_pil).astype(np.float32) / 255.0

    _debug_save_image(out, "final_composite_rgba")

    # ======================================================
    # WRITE BACK TO BLENDER
    # ======================================================
    final_img.pixels[:] = out.reshape(-1)
    final_img.update()

    print("✔ Composite complete (Photoshop-style RGBA dilation).")

def normalize_lightmap_for_png(img):
    """
    Rescale linear HDR lightmap so max RGB == 1.0
    Preserves color ratios, PNG-safe.
    """
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)

    rgb = arr[:, :, :3]

    max_val = np.max(rgb)

    if max_val > 1.0:
        rgb /= max_val
        arr[:, :, :3] = rgb

    img.pixels[:] = arr.reshape(-1)
    img.update()

    return max_val  # keep this if you want to store scale

def bake_lightmap_async(obj, lightmap_group, resolution=2048, callback=None):
    bake_queue.append((obj, lightmap_group, resolution, callback))
    if not bpy.app.timers.is_registered(_process_bake_queue):
        bpy.app.timers.register(_process_bake_queue, first_interval=0.01)

def queue_group_bakes(group_items, resolution, callback, on_complete=None):
    global group_pipeline_stage
    global group_pipeline_index
    global group_pipeline_jobs
    global group_bake_active
    global group_pipeline_complete_callback

    group_pipeline_complete_callback = on_complete

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

def rebuild_groups_after_pipeline(finished_jobs):
    """
    Automatically rebuild my_items list using the separated objects
    returned from the combine→bake→separate pipeline.
    """
    scene = bpy.context.scene

    # Clear existing groups
    scene.my_items.clear()
    global baked_object
    baked_object = []

    for i, job in enumerate(finished_jobs):
        sep_objects = job.get("separated_objects")
        print(sep_objects)

        if not sep_objects:
            continue  # skip empty jobs

        # Make new group entry
        group = scene.my_items.add()
        group.name = f"Lightgroup {len(scene.my_items)}"

        for name in sep_objects:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            if obj.as_pointer() == 0:
                continue

            entry = group.objects.add()
            entry.obj = obj
            baked_object.append(obj)
    
    props = bpy.context.scene.lgx_preview_props
    props.disable_preview_on = False

    print("✔ Rebuilt groups after pipeline")

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

def smart_uv_unwrap(mesh_object):
    # Must be in edit mode, mesh selected
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66),   # 66° like the UI
        island_margin=0.001,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )

    bpy.ops.object.mode_set(mode='OBJECT')

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

def view3d_override():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                            'scene': bpy.context.scene,
                            'view_layer': bpy.context.view_layer,
                        }
    # fallback
    return bpy.context.copy()


            


###########
#   Combine / seperate
###########

def combine(selected_objects, ops=None):
    ctx = view3d_override()

    with bpy.context.temp_override(**ctx):
        for mesh_object in selected_objects:
            make_object_active_and_selected(mesh_object)

            # Switch to EDIT and select all
            ensure_object_mode_is('EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

            # Back to OBJECT
            ensure_object_mode_is('OBJECT')

            # Build vertex group name
            real_mat = mesh_object.active_material
            if real_mat:
                material_name = real_mat.name
            else:
                material_name = "Material"

            vertex_group_name = f"{mesh_object.name}{VERTEX_GROUP_SPLITTER}{material_name}"

            print(f"VERTEX_GROUP {vertex_group_name}")

            vertex_group_reference = mesh_object.vertex_groups.get(vertex_group_name)
            if vertex_group_reference is None:
                vertex_group_reference = mesh_object.vertex_groups.new(name=vertex_group_name)

            # Add all vertices to group
            all_vertex_indices = [v.index for v in mesh_object.data.vertices]
            if all_vertex_indices:
                vertex_group_reference.add(all_vertex_indices, 1.0, 'ADD')

        # Join phase
        bpy.ops.object.select_all(action="DESELECT")
        for mesh_object in selected_objects:
            mesh_object.select_set(True)

        bpy.context.view_layer.objects.active = selected_objects[0]

        bpy.ops.object.join()

        active_obj = bpy.context.view_layer.objects.active

    return active_obj

def seperate(active_object: bpy.types.Object, ops=None):
    ctx = view3d_override()

    with bpy.context.temp_override(**ctx):
        active_object.name = "SPLITTING OBJECT"

        if active_object is None or active_object.type != "MESH":
            if ops is not None:
                ops.report({'ERROR'}, "Select a mesh that contains vertex groups.")
            return {'CANCELLED'}

        ensure_object_mode_is('EDIT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        ensure_object_mode_is('OBJECT')

        vertex_groups_to_process = []
        for vg in active_object.vertex_groups:
            vertex_groups_to_process.append(vg)

        og_object_list = []
        for vertex_group in vertex_groups_to_process:
            vertex_group_name = vertex_group.name
            if VERTEX_GROUP_SPLITTER not in vertex_group_name:
                continue

            original_object_name, original_material_name = vertex_group_name.split(VERTEX_GROUP_SPLITTER)

            # Deselect all
            ensure_object_mode_is('EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            ensure_object_mode_is('OBJECT')

            # Select vertices belonging to group
            vertex_indices_in_group = []
            for vertex in active_object.data.vertices:
                for vertex_group_link in vertex.groups:
                    if vertex_group_link.group == vertex_group.index:
                        vertex_indices_in_group.append(vertex.index)

            for idx in vertex_indices_in_group:
                active_object.data.vertices[idx].select = True

            active_object.data.update()

            ensure_object_mode_is('EDIT')
            bpy.ops.mesh.separate(type='SELECTED')
            ensure_object_mode_is('OBJECT')

            newly_created_objects = []
            for o in bpy.context.selected_objects:
                if o != active_object:
                    newly_created_objects.append(o)

            if not newly_created_objects:
                print("WARNING: mesh.separate() created no new object for", vertex_group_name)
                continue

            new_obj = newly_created_objects[-1]

            # Remove existing conflicting object
            existing = bpy.data.objects.get(original_object_name)
            if existing is not None and existing != new_obj:
                bpy.data.objects.remove(existing, do_unlink=True)

            new_obj.name = original_object_name

            new_obj.data.materials.clear()
            # First try to find exact match
            mat = bpy.data.materials.get(original_material_name)

            # If not found, try unique variant
            if mat is None:
                unique_name = f"{original_material_name}_unique_{original_object_name}"
                mat = bpy.data.materials.get(unique_name)

            if mat:
                new_obj.data.materials.append(mat)

            new_obj.vertex_groups.clear()


            og_object_list.append(new_obj.name)

        bpy.data.objects.remove(active_object, do_unlink=True)

        if ops is not None:
            ops.report({'INFO'}, "Meshes successfully restored.")

    return og_object_list

###########
#   preview
###########

def preview_object(object: bpy.types.Object, preview=True):
    mat = object.active_material
    if not mat:
        return
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = None
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n
            break


    base_input = bsdf.inputs["Base Color"]
    emit_input = bsdf.inputs["Emission Color"]

    # If nothing is linked to Base Color, do nothing
    if not base_input.is_linked:
        return

    # The current link into BSDF Base Color
    old_link = base_input.links[0]
    from_sock = old_link.from_socket
    linked_node = from_sock.node

    # ----------------------------------------------------------
    # CASE 1: preview=False AND BaseColor is already MixRGB
    # → REMOVE the MixRGB and restore original image
    # ----------------------------------------------------------
    if linked_node.type == "MIX_RGB" and preview == False:
        mix_node = linked_node

        # Get the original color1 texture socket
        orig_sock = mix_node.inputs["Color1"].links[0].from_socket

        # Remove mix node connection to BSDF
        links.remove(old_link)

        # Restore original texture directly to Base Color
        links.new(orig_sock, base_input)

        # Optional: delete the mix node
        nodes.remove(mix_node)

        return  # restoration complete


    # ----------------------------------------------------------
    # CASE 2: preview=True AND BaseColor is a plain Image Texture
    # → INSERT a MixRGB that uses the Emission texture
    # ----------------------------------------------------------
    if linked_node.type == "TEX_IMAGE" and preview == True:

        # Create MixRGB
        mix_RGB = nodes.new("ShaderNodeMixRGB")
        mix_RGB.location = (
            (linked_node.location.x + bsdf.location.x) / 2,
            linked_node.location.y - 80
        )

        # Mix mode = Multiply
        mix_RGB.blend_type = "MULTIPLY"

        # Factor = 1.0
        mix_RGB.inputs["Fac"].default_value = 1.0

        # Remove direct link (image → bsdf)
        links.remove(old_link)

        # Connect base texture → Color1
        links.new(from_sock, mix_RGB.inputs["Color1"])

        # If emission is linked, plug into Color2
        if emit_input.is_linked:
            em_link = emit_input.links[0]
            em_node = em_link.from_socket.node

            if em_node.type == "TEX_IMAGE":
                links.new(em_link.from_socket, mix_RGB.inputs["Color2"])

        # Output of MixRGB → BSDF Base Color
        links.new(mix_RGB.outputs["Color"], base_input)

def rebuild_baked_mat_list():
    all_objects = bpy.context.selectable_objects

    global baked_object
    baked_object.clear()

    for obj in all_objects:
        if obj.type == "MESH":

            mat = obj.active_material
            if not mat:
                return
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            bsdf = None
            for n in nodes:
                if n.type == "BSDF_PRINCIPLED":
                    bsdf = n
                    break
            
            emit_input = bsdf.inputs["Emission Color"]

            if emit_input.is_linked:
                em_link = emit_input.links[0]
                em_node = em_link.from_socket.node

                if em_node.type == "TEX_IMAGE":
                    img = em_node.image
                    try:
                        filename = os.path.basename(img.filepath)

                        if "lightmap_Lightgroup" in filename:
                            baked_object.append(obj)
                    except:
                        pass

def preview_init_check():
    rebuild_baked_mat_list()
    global baked_object


    props = bpy.context.scene.lgx_preview_props
    if len(baked_object) != 0:
        mat = baked_object[0].active_material
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = None
        for n in nodes:
            if n.type == "BSDF_PRINCIPLED":
                bsdf = n
                break


        base_input = bsdf.inputs["Base Color"]

        # If nothing is linked to Base Color, do nothing
        if not base_input.is_linked:
            return

        # The current link into BSDF Base Color
        old_link = base_input.links[0]
        from_sock = old_link.from_socket
        linked_node = from_sock.node


        if linked_node.type == "MIX_RGB":
            props.disable_preview_off = False   
            props.disable_preview_on = True
        elif linked_node.type == "TEX_IMAGE":
            props.disable_preview_off = True   
            props.disable_preview_on = False
    
    
    
    else:
        props.disable_preview_off = True   
        props.disable_preview_on = True

def remove_lightmaps():
    global baked_object
    if len(baked_object) == 0:
        return {'FINISHED'}
    
    for obj in baked_object:
        preview_object(obj, preview=False)

        mat = obj.active_material
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = None
        for n in nodes:
            if n.type == "BSDF_PRINCIPLED":
                bsdf = n
                break


        emit_input = bsdf.inputs["Emission Color"]

        if emit_input.is_linked:
            em_link = emit_input.links[0]
            em_node = em_link.from_socket.node

            if em_node.type == "TEX_IMAGE":
                img = em_node.image
                filename = os.path.basename(img.filepath)

                if "lightmap_Lightgroup" in filename:
                    # Remove the emission link first
                    links.remove(em_link)

                    # Collect nodes to delete
                    to_delete = set()
                    stack = [em_node]

                    # Walk backwards through the graph
                    while stack:
                        node = stack.pop()
                        if node in to_delete:
                            continue
                        
                        to_delete.add(node)

                        for inp in node.inputs:
                            if inp.is_linked:
                                prev_node = inp.links[0].from_socket.node
                                stack.append(prev_node)

                    # Delete nodes
                    for node in to_delete:
                        nodes.remove(node)

                    # Remove orphaned image
                    if img and img.users == 0:
                        bpy.data.images.remove(img)

def ensure_unique_material(obj):
    """
    If the object's active material is used by other objects,
    duplicate it and assign the duplicate ONLY to this object.
    """

    if not obj or not obj.material_slots:
        return

    mat = obj.active_material
    if not mat:
        return

    # Find all objects using this material
    shared_by = [
        o for o in bpy.data.objects
        if o.type == 'MESH'
        and o.material_slots
        and any(slot.material == mat for slot in o.material_slots)
    ]

    # If only this object uses it, nothing to do
    if len(shared_by) <= 1:
        return

    # Duplicate material
    new_mat = mat.copy()

    # Assign to this object
    obj.active_material = new_mat

    print(f"✔ {obj.name} now uses its own material: {new_mat.name}")




class LGX_bake_samples(bpy.types.PropertyGroup):

    def update_samples(self, context):
        # Map enum → numeric samples
        SAMPLES_MAP = {
            "LOW": 32,
            "MEDIUM": 64,
            "NORMAL": 128,
            "HIGH": 256,
            "VERY_HIGH": 512,
            "ULTRA": 1024,
        }
        self.value = SAMPLES_MAP[self.mode]

    mode: bpy.props.EnumProperty(
        name="Samples",
        description="Select number of Cycles samples for baking",
        items=[
            ("LOW", "32", "Low samples, fastest (may have noise)"),
            ("MEDIUM", "64", "Medium samples, good balance"),
            ("NORMAL", "128", "Normal samples (default, recommended)"),
            ("HIGH", "256", "High samples, better quality"),
            ("VERY_HIGH", "512", "Very high samples, excellent quality"),
            ("ULTRA", "1024", "Ultra samples, best quality (slowest)"),
        ],
        default="NORMAL",
        update=update_samples,
    )# type: ignore

    # Stores actual numeric samples
    value: bpy.props.IntProperty(
        name="Samples Value",
        default=128
    )# type: ignore

class LGX_PreviewProps(bpy.types.PropertyGroup):
    disable_preview_on: bpy.props.BoolProperty(
        name="Disable Preview On",
        default=True
    ) # type:ignore

    disable_preview_off: bpy.props.BoolProperty(
        name="Disable Preview Off",
        default=True
    ) # type:ignore

class LGX_ObjectRef(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(type=bpy.types.Object) # type: ignore

class LGX_ListItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Item") # type: ignore
    objects: bpy.props.CollectionProperty(type=LGX_ObjectRef) # type: ignore
    objects_index: bpy.props.IntProperty(default=0) # type: ignore

class LGX_group_generator_count(bpy.types.PropertyGroup):

    count: bpy.props.IntProperty(
        name="",
        description="",
        default=2,
        min=0,
        max=1000
    ) # type: ignore

class LGX_resolution(bpy.types.PropertyGroup):

    def update_resolution(self, context):
        # Map enum → numeric resolution
        RES_MAP = {
            "LOW": 512,
            "MEDIUM": 1024,
            "HIGH": 2048,
        }
        self.value = RES_MAP[self.mode]

    mode: bpy.props.EnumProperty(
        name="Resolution",
        description="Select lightmap resolution",
        items=[
            ("LOW", "512", "Low resolution, fastest"),
            ("MEDIUM", "1024", "Medium resolution (recommended)"),
            ("HIGH", "2048", "Best quality but slower"),
        ],
        default="MEDIUM",
        update=update_resolution,
    )# type: ignore

    # Stores actual numeric resolution
    value: bpy.props.IntProperty(
        name="Resolution Value",
        default=1024
    )# type: ignore

class LGX_render_light_level(bpy.types.PropertyGroup):

    def update_light_level(self, context):
        # Map enum → numeric light level
        LEVEL_MAP = {
            "LOW": 3.5,
            "MEDIUM": 4.5,
            "HIGH": 6.0,
        }
        self.value = LEVEL_MAP[self.mode]

    mode: bpy.props.EnumProperty(
        name="Render Light Level",
        description="Maximum light level used for Map Range normalization",
        items=[
            ("LOW", "3.5", "Low (indoor / soft lighting)"),
            ("MEDIUM", "4.5", "Medium (balanced lighting)"),
            ("HIGH", "6.0", "High (bright / strong lights)"),
        ],
        default="LOW",   # 3.5 default
        update=update_light_level,
    )  # type: ignore

    # Stores actual numeric light level
    value: bpy.props.FloatProperty(
        name="Light Level Value",
        default=3.5
    )  # type: ignore


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

class LGX_OT_UV_uwrap_uvs(bpy.types.Operator):
    bl_idname = "lgx._uwrap_uvs"
    bl_label = "unwrap UVS"

    def execute(self, context):
        obj = context.selected_objects[0]
        smart_uv_unwrap(obj)

        return {"FINISHED"}

#preview
class LGX_OT_preview_on(bpy.types.Operator):
    bl_idname = "lgx.preview_on"
    bl_label = "unwrap UVS"

    def execute(self, context):
        global baked_object
        print(baked_object)

        if len(baked_object) == 0:
            print("attempting to rebuild baked groups")
            rebuild_baked_mat_list()
            if len(baked_object) == 0:
                return {"FINISHED"}

        props = context.scene.lgx_preview_props
        props.disable_preview_off = False   # disable OFF button
        props.disable_preview_on = True

        for obj in baked_object:
            preview_object(obj, preview=True)
        
        return {"FINISHED"}

class LGX_OT_preview_off(bpy.types.Operator):
    bl_idname = "lgx.preview_off"
    bl_label = "unwrap UVS"

    def execute(self, context):
        global baked_object
        print(baked_object)

        if len(baked_object) == 0:
            print("attempting to rebuild baked groups")
            rebuild_baked_mat_list()
            if len(baked_object) == 0:
                return {"FINISHED"}
        
        props = context.scene.lgx_preview_props
        props.disable_preview_off = True   # disable OFF button
        props.disable_preview_on = False

        for obj in baked_object:
            preview_object(obj, preview=False)

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
        seperate(active_object, ops=self)
        return {"FINISHED"}


#Random select
class LGX_OT_lightmap_groups_generator(bpy.types.Operator):
    bl_idname = "lgx.lightmap_groups_generator"
    bl_label = "generate light map groups"

    def execute(self, context):
        selectable_objects = bpy.context.selected_objects

        print(selectable_objects)

        max_per = bpy.context.scene.lgx_group_generator_count.count
        print(max_per)

        object_list:list[bpy.types.Object] = []

        final_list:list[list[bpy.types.Object]] = []

        for object in selectable_objects:
            if object.type == "MESH" and len(object.data.polygons) > 0:
                object_list.append(object)

        if len(object_list) <= max_per:
            final_list.append(object_list)

        print(object_list)

        light_pack_list = []
        for object in object_list:

            if len(light_pack_list) < max_per:
                light_pack_list.append(object)
            
            else:
                final_list.append(light_pack_list)
                light_pack_list = []
                light_pack_list.append(object)
        
        final_list.append(light_pack_list)
        
        print(final_list)

        scene = context.scene

        for obj_list in final_list:
            item = scene.my_items.add()
            item.name = "Lightgroup " + str(len(scene.my_items))

            for obj in obj_list:
                entry = item.objects.add()
                entry.obj = obj
        
        
        return {"FINISHED"}

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

        rebuild_baked_mat_list()
        remove_lightmaps()


        for group in scene.my_items:
            for ref in group.objects:
                obj = ref.obj
                ensure_unique_material(obj)



        def save_lightmap(img, lightmap_group):

            base_path = bpy.path.abspath(context.scene.lgx_bake_settings.LGX_bake_output_folder)


            path = os.path.join(base_path, f"lightmap_{lightmap_group}.png")

            img.filepath_raw = path
            img.file_format = 'PNG'
            img.save()
            print(f"Saved lightmap to: {path}")
            

        queue_group_bakes(
            scene.my_items,
            resolution=context.scene.lgx_render_resolution.value,
            callback=save_lightmap,
            on_complete=rebuild_groups_after_pipeline
        )

        return {'FINISHED'}

class LGX_remove_lightmaps(bpy.types.Operator):
    bl_idname = "lgx.remove_lightmap"
    bl_label = "remove lightmaps"

    def execute(self, context):
        
        remove_lightmaps()

        bpy.ops.outliner.orphans_purge()

        return {'FINISHED'}
    
    

#debug preview
class LGX_preview_on(bpy.types.Operator):
    bl_idname = "lgx.dbg_preview_on"
    bl_label = "turn on preview on object"


    def execute(self, context):
        active_object = context.active_object

        preview_object(object=active_object, preview=True)

        return {'FINISHED'}
    
class LGX_preview_off(bpy.types.Operator):
    bl_idname = "lgx.dbg_preview_off"
    bl_label = "turn on preview on object"


    def execute(self, context):
        active_object = context.active_object

        preview_object(object=active_object, preview=False)

        return {'FINISHED'}

class LGX_ensure_unique(bpy.types.Operator):
    bl_idname = "lgx.ensure_unique"
    bl_label = "uwu"
    def execute(self, context):
        objects:list[bpy.types.Object] = bpy.context.selected_objects


        for obj in objects:
            ensure_unique_material(obj)
            
            print(obj.active_material.name)

        return {"FINISHED"}

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
        props = context.scene.lgx_preview_props

        box = layout.box()
        box.label(text="Objects", icon='MESH_PLANE')

        row = box.row()
        box.operator("lgx.lightmap_groups_generator", icon="ZOOM_SELECTED", text="Generate Groups")
        box.prop(context.scene.lgx_group_generator_count, "count")


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

        box = layout.box()
        row = box.row()
        

        innerbox = box.box()
        innerbox.label(text="Bake Output location")
        innerbox.prop(context.scene.lgx_bake_settings, "LGX_bake_output_folder")

        innerbox = box.box()
        innerbox.label(text="Lightmap Resolution")
        innerbox.prop(context.scene.lgx_render_resolution, "mode")

        innerbox = box.box()
        innerbox.label(text="Bake Samples")
        innerbox.prop(context.scene.lgx_bake_samples, "mode")

        innerbox = box.box()
        innerbox.label(text="Render Light Level")
        innerbox.prop(context.scene.lgx_render_light_level, "mode")

        box.operator("lgx.bake", text="Bake Lightmap")

        row = box.row()

                # --- Preview ON ---
        on_col = row.column()
        on_col.enabled = not props.disable_preview_on
        on_col.operator("lgx.preview_on", text="Preview ON")

        # --- Preview OFF ---
        off_col = row.column()
        off_col.enabled = not props.disable_preview_off
        off_col.operator("lgx.preview_off", text="Preview OFF")

        box.operator("lgx.remove_lightmap", text="Reset")

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
        layout.separator()
        layout.operator("lgx._uwrap_uvs", text = "unwrap selected objects")

        layout.label(text=f"current selected group {context.scene.my_items_index}")

        layout.label(text=f"render res = {context.scene.lgx_render_resolution.value}")

        layout.operator("lgx.ensure_unique", text = "unique")
        box = layout.box()

        box.label(text="preview_objects")
        box.operator("lgx.dbg_preview_on", text="Preview On")
        box.operator("lgx.dbg_preview_off", text="Preview Off")





classes = [
    LGX_render_light_level,
    LGX_bake_samples,
    LGX_ensure_unique,
    LGX_remove_lightmaps,
    LGX_PreviewProps,
    LGX_OT_preview_on,
    LGX_OT_preview_off,
    LGX_preview_off,
    LGX_preview_on,
    LGX_resolution,
    LGX_ObjectRef,
    LGX_group_generator_count,
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

    LGX_OT_lightmap_groups_generator,

    LGX_PT_panel,
]

debug_classes = [
    LGX_PT_debug,
    LGX_OT_UV_uwrap_uvs
]

if DEBUG == True:
    classes.extend(debug_classes)

def register():
    global baked_object
    for class_type in classes:
        bpy.utils.register_class(class_type)

    bpy.types.Scene.lgx_bake_settings = bpy.props.PointerProperty(type=LGX_bake_output_file)
    bpy.types.Scene.my_items = bpy.props.CollectionProperty(type=LGX_ListItem)
    bpy.types.Scene.my_items_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.lgx_bake_samples = bpy.props.PointerProperty(type=LGX_bake_samples)
    bpy.types.Scene.lgx_group_generator_count = bpy.props.PointerProperty(type=LGX_group_generator_count)
    bpy.types.Scene.lgx_render_resolution = bpy.props.PointerProperty(type=LGX_resolution)
    bpy.types.Scene.lgx_preview_props = bpy.props.PointerProperty(type=LGX_PreviewProps)
    bpy.types.Scene.lgx_render_light_level = bpy.props.PointerProperty(type=LGX_render_light_level)

    preview_init_check()

def unregister():
    del bpy.types.Scene.lgx_bake_settings
    del bpy.types.Scene.my_items
    del bpy.types.Scene.my_items_index    
    del bpy.types.Scene.lgx_bake_samples
    del bpy.types.Scene.lgx_group_generator_count
    del bpy.types.Scene.lgx_render_resolution
    del bpy.types.Scene.lgx_preview_props
    del bpy.types.Scene.lgx_render_light_level

    for class_type in reversed(classes):
        bpy.utils.unregister_class(class_type)


if __name__ == "__main__":
    register()