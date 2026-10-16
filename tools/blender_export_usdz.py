import bpy, sys
out = sys.argv[sys.argv.index("--") + 1]
rig = bpy.data.objects["rig"]
keep = {rig} | {o for o in bpy.data.objects if o.parent == rig and o.type == "MESH"}
for o in list(bpy.data.objects):
    if o not in keep:
        bpy.data.objects.remove(o, do_unlink=True)

for pb in rig.pose.bones:
    pb.matrix_basis.identity()

for c in bpy.data.collections:
    c.hide_viewport = False; c.hide_render = False
def unexclude(lc):
    lc.exclude = False; lc.hide_viewport = False
    for ch in lc.children: unexclude(ch)
unexclude(bpy.context.view_layer.layer_collection)
root = bpy.context.scene.collection
for o in keep:
    if o.name not in root.objects: root.objects.link(o)
    o.hide_set(False); o.hide_viewport = False; o.hide_render = False
print("EXPORT rig visible:", rig.visible_get(), "collections:", [c.name for c in rig.users_collection])
print("EXPORT objects:", sorted(o.name for o in keep))

bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = rig.data.edit_bones
def def_equiv(name):
    base = name.split("-", 1)[1] if "-" in name else name
    cand = eb.get("DEF-" + base)
    return cand if (cand is not None and cand.use_deform) else None
reparented = 0
for b in list(eb):
    if not b.use_deform or b.parent is None or b.parent.use_deform:
        continue
    p = b.parent
    target = None
    while p is not None:
        if p.use_deform: target = p; break
        t = def_equiv(p.name)
        if t is not None and t != b: target = t; break
        p = p.parent
    if target is not None:
        b.use_connect = False
        b.parent = target
        reparented += 1
bpy.ops.object.mode_set(mode="OBJECT")
print("EXPORT reparented", reparented)
deform = [b.name for b in rig.data.bones if b.use_deform]
print("EXPORT deform bones:", len(deform))
for b in rig.data.bones:
    if b.use_deform and any(k in b.name for k in ("spine", "thigh", "shin", "foot", "toe", "shoulder", "upper_arm", "forearm", "hand.", "pelvis")):
        print(f"  BONE {b.name} parent={b.parent.name if b.parent else None} head={tuple(round(v,3) for v in b.head_local)}")
bpy.ops.wm.usd_export(
    filepath=out, export_armatures=True, only_deform_bones=True, export_shapekeys=False,
    export_animation=False, export_materials=True, export_textures_mode="NEW", relative_paths=True,
    convert_orientation=True, export_global_forward_selection="NEGATIVE_Z", export_global_up_selection="Y",
    selected_objects_only=False, generate_materialx_network=False,
    convert_world_material=False, convert_scene_units="METERS",
)
print("EXPORT done", out)
