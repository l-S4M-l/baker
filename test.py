import bpy


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


objects:list[bpy.types.Object] = bpy.context.selected_objects


for obj in objects:
    ensure_unique_material(obj)
    
    print(obj.active_material.name)