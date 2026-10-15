#!/bin/zsh
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate gemx
W=/Users/anuraag/Developer/GEMX_ONNX; cd /Users/anuraag/Developer/GEM-X
REF=$W/runs/visual/test3/pytorch_postproc_hpe_results.pt
for v in test3_1080p_10fps test3_1080p_15fps test3_540p_30fps test3_270p_30fps test3_540p_10fps; do
  echo "=== $v $(date +%H:%M:%S)"; python $W/tests/t7_fpsres.py --video $W/runs/fpsres/$v.mp4 --ref $REF --ref-fps 29.97 2>&1 | grep -E "^\[t7\]|Traceback|Error" | cut -c1-900; echo "=== $v exit=$? $(date +%H:%M:%S)"
done
echo "=== quant_extra $(date +%H:%M:%S)"; python $W/tests/t7_quant_extra.py 2>&1 | grep -E "^\[t7b\]|Traceback|Error" | cut -c1-900; echo "=== quant_extra exit=$? $(date +%H:%M:%S)"
echo ALL DONE
