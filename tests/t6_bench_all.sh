#!/bin/zsh
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate gemx
W=/Users/anuraag/Developer/GEMX_ONNX; S=/Users/anuraag/Developer/GEM-X/sample_inputs
cd /Users/anuraag/Developer/GEM-X
VIDS=("$@"); (( $#VIDS == 0 )) && VIDS=(test3_demo_6fps test3_demo_8fps test2_demo_6fps test2_demo_8fps test3 test2 test)
for v in $VIDS; do
  echo "=== $v $(date +%H:%M:%S)" | tee -a $W/runs/bench/_driver.log
  /usr/bin/time -l python $W/tests/t6_bench.py --video $S/$v.mp4 > $W/runs/bench/$v.log 2>&1
  echo "=== $v exit=$? $(date +%H:%M:%S)" | tee -a $W/runs/bench/_driver.log
  grep "maximum resident" $W/runs/bench/$v.log | tee -a $W/runs/bench/_driver.log
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a $W/runs/bench/_driver.log
