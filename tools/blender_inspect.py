import bpy
print("BLENDER", bpy.app.version_string)
for ob in bpy.data.objects:
    if ob.type == "ARMATURE":
        bones = ob.data.bones
        deform = [b.name for b in bones if b.use_deform]
        print(f"ARMATURE {ob.name}: {len(bones)} bones, {len(deform)} deform; scale={tuple(round(s,3) for s in ob.scale)}")
        for b in bones:
            if b.use_deform:
                print(f"  DEF {b.name} parent={b.parent.name if b.parent else None} head={tuple(round(v,3) for v in b.head_local)} tail={tuple(round(v,3) for v in b.tail_local)}")
for ob in bpy.data.objects:
    if ob.type == "MESH":
        mods = [(m.type, getattr(m, 'object', None) and m.object.name) for m in ob.modifiers]
        vg = len(ob.vertex_groups)
        weighted = sum(1 for v in ob.data.vertices if len(v.groups) > 0)
        print(f"MESH {ob.name}: verts={len(ob.data.vertices)} vgroups={vg} weighted={weighted} parent={ob.parent.name if ob.parent else None} mods={mods} dims={tuple(round(d,3) for d in ob.dimensions)}")
