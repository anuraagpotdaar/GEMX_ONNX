import numpy as np, time, coremltools as ct, sys
sys.path.insert(0,'/Users/anuraag/Developer/GEMX_ONNX/phone'); import gemx_phone as gp
import coremltools.optimize.coreml as cto
Q='/Users/anuraag/Developer/GEMX_ONNX/quant/'
R=np.load('/Users/anuraag/Developer/GEMX_ONNX/runs/bench/vitpose_ref_crops.npz'); crops=R['crops']; hm_ref=R['hm_ref']; bb=R['bb']; kp_ref=R['kp_ref']; N=len(crops)
def kps(hm): return np.stack([gp.heatmaps_to_kp2d(hm[i:i+1], bb[i:i+1])[0] for i in range(N)])
k_ref=kps(hm_ref)
def report(name, hm, t):
    k=kps(hm); e=np.linalg.norm(k[...,:2]-k_ref[...,:2],axis=-1); eref=np.linalg.norm(k[...,:2]-kp_ref[...,:2],axis=-1)
    print(f'{name:40s} {t*1000:7.1f} ms/frame | kp vs fp32-noflip: mean {e.mean():.2f} px max {e.max():.1f} | vs MPS flip-test ref: mean {eref.mean():.2f} px', flush=True)
def bench(path, cu, name, flip=False):
    t0=time.time(); m=ct.models.MLModel(path, compute_units=cu); tl=time.time()-t0
    t0=time.time(); m.predict({'imgs':crops[:1]}); tf=time.time()-t0
    t0=time.time(); hms=[]
    for i in range(N):
        hm=m.predict({'imgs':crops[i:i+1]})['heatmaps'].astype(np.float32)
        if flip:
            hf=m.predict({'imgs':np.ascontiguousarray(crops[i:i+1][:,:,:,::-1])})['heatmaps'].astype(np.float32)
            hm=(hm+gp.flip_heatmaps(hf))*0.5 if hasattr(gp,'flip_heatmaps') else (hm+flip_hm(hf))*0.5
        hms.append(hm)
    t=(time.time()-t0)/N; print(f'  [{name}] load {tl:.1f} s, first {tf:.1f} s'); report(name, np.concatenate(hms), t); del m

pairs=[(9,10),(11,39),(12,40),(13,41),(14,42),(15,43),(16,44),(17,45),(18,46),(19,47),(20,48),(21,49),(22,50),(23,51),(24,52),(25,53),(26,54),(27,55),(28,56),(29,57),(30,58),(31,59),(32,60),(33,61),(34,62),(35,63),(36,64),(37,65),(38,66),(67,72),(68,73),(69,74),(70,75),(71,76)]
def flip_hm(h):
    y=h.copy()
    for l,r in pairs: y[:,l]=h[:,r]; y[:,r]=h[:,l]
    return y[:,:,:,::-1]
bench(Q+'vitpose_fp16.mlpackage', ct.ComputeUnit.CPU_AND_NE, 'fp16 CPU_AND_NE')
bench(Q+'vitpose_fp16.mlpackage', ct.ComputeUnit.ALL, 'fp16 ALL + flip test', flip=True)

m=ct.models.MLModel(Q+'vitpose_fp16.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
t0=time.time()
cfg=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8', granularity='per_channel'))
mq=cto.linear_quantize_weights(m, config=cfg); mq.save(Q+'vitpose_w8.mlpackage'); print('w8 saved', round(time.time()-t0), 's', flush=True); del m, mq
import subprocess; print(subprocess.run(['du','-sh',Q+'vitpose_fp16.mlpackage',Q+'vitpose_w8.mlpackage'],capture_output=True,text=True).stdout)
bench(Q+'vitpose_w8.mlpackage', ct.ComputeUnit.ALL, 'w8 ALL')
bench(Q+'vitpose_w8.mlpackage', ct.ComputeUnit.CPU_AND_NE, 'w8 CPU_AND_NE')
bench(Q+'vitpose_w8.mlpackage', ct.ComputeUnit.CPU_ONLY, 'w8 CPU only')
print('done')
