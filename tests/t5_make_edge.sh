#!/bin/zsh
set -e
G=/Users/anuraag/Developer/GEM-X; E=/Users/anuraag/Developer/GEMX_ONNX/runs/edge; mkdir -p $E
ffmpeg -y -loglevel error -i $G/inputs/videos/person.mp4 -frames:v 8 -c:v libx264 -crf 18 -pix_fmt yuv420p $E/short8.mp4
ffmpeg -y -loglevel error -f lavfi -i color=c=black:s=432x768:r=10 -frames:v 30 -c:v libx264 -crf 18 -pix_fmt yuv420p $E/black30.mp4
ffmpeg -y -loglevel error -i $G/sample_inputs/test3.mp4 -t 2 -c:v libx264 -crf 18 -pix_fmt yuv420p $E/test3_2s_30fps.mp4
ffmpeg -y -loglevel error -i $G/inputs/videos/person.mp4 -frames:v 32 -c:v libx264 -crf 18 -pix_fmt yuv420p $E/person_32f.mp4
echo "not a video" > $E/notes.txt
for f in $E/*.mp4; do echo -n "$f: "; ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 $f; done
