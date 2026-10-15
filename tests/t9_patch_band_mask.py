import sys, onnx
from onnx import helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
prod = {o: n for n in g.node for o in n.output}

targets = []
for sm in [n for n in g.node if n.op_type == "Softmax"]:
    w = prod.get(sm.input[0]); assert w is not None and w.op_type == "Where", (sm.name, w.op_type if w else None); targets.append(w)
assert targets, "no attention Where nodes found"
g.input.append(helper.make_tensor_value_info("attn_mask", TensorProto.BOOL, ["L", "L"]))
new_nodes = []
for n in targets:
    cond = n.input[0]; p = prod[cond]; assert p.op_type in ("Cast", "Expand", "Reshape"), (n.name, p.op_type)
    out = cond + "_band_" + str(len(new_nodes)); new_nodes.append((n, helper.make_node("Or", [cond, "attn_mask"], [out], name=n.name + "_band_or"))); n.input[0] = out

nodes = list(g.node); del g.node[:]
for n in nodes:
    for w, o in new_nodes:
        if w is n: g.node.append(o)
    g.node.append(n)
onnx.save(m, dst, save_as_external_data=False)
print(f"[t9] patched {len(targets)} attention blocks -> {dst}")
