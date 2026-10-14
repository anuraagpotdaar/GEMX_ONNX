#!/bin/zsh
set -u
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate gemx
API=/Users/anuraag/Developer/BAMM_API; W=/Users/anuraag/Developer/GEMX_ONNX; R=$W/runs/service; mkdir -p $R
PORT=${PORT:-8041}
cd $API && SERVICE=gemx GEMX_ROOT=/Users/anuraag/Developer/GEM-X PORT=$PORT OUTPUTS_DIR=$R/outputs GEMX_KEEP_WARM=${GEMX_KEEP_WARM:-0} \
  python services/csgpu_serve.py > $R/service.log 2>&1 &
SVC=$!
echo "service pid $SVC"; for i in {1..120}; do curl -sf http://127.0.0.1:$PORT/ >/dev/null && break; sleep 2; done
curl -s http://127.0.0.1:$PORT/ | head -c 600; echo
post() { # name video extra-form-args...
  local name=$1 video=$2; shift 2
  local t0=$(date +%s)
  local code=$(curl -s -o $R/$name.bvh -w '%{http_code}' -D $R/$name.headers -F "video=@$video" "$@" http://127.0.0.1:$PORT/generate)
  echo "$name: http=$code t=$(( $(date +%s) - t0 ))s bytes=$(wc -c < $R/$name.bvh) $(grep -i 'x-frames\|x-fps' $R/$name.headers | tr -d '\r' | tr '\n' ' ')"
}
G=/Users/anuraag/Developer/GEM-X
post person_pytorch        $G/inputs/videos/person.mp4 -F denoiser=pytorch -F job_id=job1
post person_onnx           $G/inputs/videos/person.mp4 -F denoiser=onnx
post person_static_noimg   $G/inputs/videos/person.mp4 -F denoiser=onnx -F static_cam=true -F no_imgfeat=true
post test3_8fps_pytorch    $G/sample_inputs/test3_demo_8fps.mp4 -F denoiser=pytorch
post bad_ext               $W/runs/edge/notes.txt
post bad_denoiser          $G/inputs/videos/person.mp4 -F denoiser=foo
[ -f $W/runs/edge/short8.mp4 ] && post short8 $W/runs/edge/short8.mp4 -F denoiser=onnx
[ -f $W/runs/edge/black30.mp4 ] && post black30 $W/runs/edge/black30.mp4 -F denoiser=onnx
curl -s http://127.0.0.1:$PORT/progress/job1; echo
kill $SVC; wait $SVC 2>/dev/null; echo "service stopped"
