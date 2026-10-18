import os, sys, time, numpy as np, torch
os.chdir('/Users/anuraag/Developer/GEM-X'); sys.path.insert(0,'/Users/anuraag/Developer/GEM-X'); sys.path.insert(0,'/Users/anuraag/Developer/GEMX_ONNX/tests')
import common as C
from gem.utils.sam3db_extractor import SAM3DBExtractor
t0=time.time(); ext=SAM3DBExtractor(device='cpu'); model=ext.estimator.model.float().eval()
if hasattr(model,'backbone_dtype'): model.backbone_dtype=torch.float32
for n,b in model.named_buffers():
    if b.dtype==torch.bfloat16: b.data=b.data.float()
print('loaded', round(time.time()-t0,1),'s', flush=True)
cfg=model.cfg; print('backbone', cfg.MODEL.BACKBONE.TYPE, 'cond', cfg.MODEL.DECODER.CONDITION_TYPE, 'kp tokens', cfg.MODEL.DECODER.get('DO_KEYPOINT_TOKENS'), 'kp3d', cfg.MODEL.DECODER.get('DO_KEYPOINT3D_TOKENS'), 'hand', cfg.MODEL.DECODER.get('DO_HAND_DETECT_TOKENS'), flush=True)
print('image_mean', model.image_mean.flatten().tolist(), 'std', model.image_std.flatten().tolist())

from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
frames=C.read_frames('/Users/anuraag/Developer/GEM-X/sample_inputs/test3_demo_6fps.mp4'); cache=C.RUNS/'bench'/'test3_demo_6fps'/'test3_demo_6fps'/'preprocess'
bbx=C.load_cache(cache)['bbx']['bbx_xys'].numpy(); cx,cy,s=bbx[0]
box=np.array([[cx-s/2,cy-s/2,cx+s/2,cy+s/2]],np.float32)
batch=prepare_batch(frames[0], ext.estimator.transform, box, masks=None, masks_score=None)
batch={k:(v if not isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
print('batch keys', {k:(tuple(v.shape) if isinstance(v,torch.Tensor) else type(v).__name__) for k,v in batch.items()}, flush=True)
model._initialize_batch(batch)
with torch.no_grad():
    out=model.forward_step(batch, decoder_type='body')
tok=out['pose_token']; print('pose_token', None if tok is None else tuple(tok.shape), flush=True)

class PoseToken(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m=m
    def forward(s, img, condition_info, ray_cond):
        emb=s.m.backbone(img, extra_embed=None)
        if isinstance(emb, tuple): emb=emb[-1]
        B=img.shape[0]
        kp=torch.zeros((B,1,3), dtype=img.dtype); kp[:,:,-1]=-2
        b={'ray_cond': ray_cond, 'cam_int': s.m._tmp_batch['cam_int'], 'img_size': s.m._tmp_batch['img_size'], 'ori_img_size': s.m._tmp_batch['ori_img_size'],
           'affine_trans': s.m._tmp_batch['affine_trans'], 'bbox_center': s.m._tmp_batch['bbox_center'], 'bbox_scale': s.m._tmp_batch['bbox_scale'], 'img': s.m._tmp_batch['img']}
        tokens,_=s.m.forward_decoder(emb, init_estimate=None, keypoints=kp, prev_estimate=None, condition_info=condition_info, batch=b)
        return tokens[:,0]
img=model.data_preprocess(model._flatten_person(batch['img']), crop_width=False)
ray=model._flatten_person(model.get_ray_condition(batch))
cond=model._get_decoder_condition(batch)
model._tmp_batch=batch
w=PoseToken(model)
with torch.no_grad(): ref=w(img, cond, ray)
t0ref=(tok[:,0] if tok.ndim==3 else tok); print('wrapper token', tuple(ref.shape), 'vs forward_step max|diff|', float((ref-t0ref).abs().max()), 'token abs mean', float(t0ref.abs().mean()), 'cos', float(torch.nn.functional.cosine_similarity(ref.flatten()[None], t0ref.flatten()[None])), flush=True)
try:

    for mod in model.modules():
        if hasattr(mod, '_jit_is_scripting'): mod._jit_is_scripting = True
    model._jit_is_scripting = True
    with torch.no_grad(): tr=torch.jit.trace(w, (img, cond, ray), check_trace=False, strict=False)
    print('TRACE OK', flush=True)

    g=tr.graph; fixed=0
    def walk(block):
        global fixed
        for n in list(block.nodes()):
            for b in n.blocks(): walk(b)
            if n.kind() in ('aten::mul','aten::add','aten::sub','aten::div','aten::rsub','aten::mul_','aten::add_'):
                for i,inp in enumerate(list(n.inputs())):
                    if str(inp.type())=='bool':
                        if inp.node().kind()=='prim::Constant':
                            c=g.create('prim::Constant'); c.f_('value', 1.0 if inp.toIValue() else 0.0); c.output().setType(torch._C.FloatType.get()); c.insertBefore(n); n.replaceInput(i, c.output())
                        else:
                            c=g.create('aten::Float', [inp]); c.output().setType(torch._C.FloatType.get()); c.insertBefore(n); n.replaceInput(i, c.output())
                        fixed+=1
    walk(g)
    print('bool constants fixed', fixed, flush=True)
    import coremltools as ct
    t0=time.time()
    ml=ct.convert(tr, inputs=[ct.TensorType(name='img', shape=tuple(img.shape)), ct.TensorType(name='condition_info', shape=tuple(cond.shape)), ct.TensorType(name='ray_cond', shape=tuple(ray.shape))],
                  outputs=[ct.TensorType(name='pose_token')], convert_to='mlprogram', compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
    ml.save('/Users/anuraag/Developer/GEMX_ONNX/quant/sam3db_posetoken_fp16.mlpackage'); print('CONVERT OK', round(time.time()-t0,1), 's', flush=True)
    m=ct.models.MLModel('/Users/anuraag/Developer/GEMX_ONNX/quant/sam3db_posetoken_fp16.mlpackage', compute_units=ct.ComputeUnit.ALL)
    feed={'img':img.numpy(),'condition_info':cond.numpy(),'ray_cond':ray.numpy()}
    o=m.predict(feed)['pose_token']; print('coreml token diff', float(np.abs(o-ref.numpy()).max()), 'cos', float(np.dot(o.flatten(),ref.numpy().flatten())/np.linalg.norm(o)/np.linalg.norm(ref.numpy())), flush=True)
    t0=time.time()
    for _ in range(5): m.predict(feed)
    print('coreml fp16 ALL ms/frame', (time.time()-t0)/5*1000, flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print('FAILED', repr(e)[:300], flush=True)
