#!/usr/bin/env bash
# Expert weights (each < 20 GB): HaMeR demo data (6.0 GB), WiLoR final checkpoint (2.6 GB) + config, SMIRK checkpoint,
# MediaPipe face landmarker. MANO / FLAME are reused from local licensed copies (not downloaded).
set -uo pipefail
E=/raid/ubx858/experts; cd $E
[ -f hamer/hamer_demo_data.tar.gz ] || curl -sL --max-time 7200 -o hamer/hamer_demo_data.tar.gz https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
echo "hamer $(stat -c %s hamer/hamer_demo_data.tar.gz)"
mkdir -p WiLoR/pretrained_models
for f in wilor_final.ckpt model_config.yaml; do [ -f WiLoR/pretrained_models/$f ] || curl -sL --max-time 7200 -o WiLoR/pretrained_models/$f https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/$f; echo "wilor $f $(stat -c %s WiLoR/pretrained_models/$f)"; done
mkdir -p smirk/pretrained_models smirk/assets
[ -f smirk/pretrained_models/SMIRK_em1.pt ] || /home/ubx858/miniconda3/envs/guava/bin/python -m gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O smirk/pretrained_models/SMIRK_em1.pt -q
echo "smirk $(stat -c %s smirk/pretrained_models/SMIRK_em1.pt 2>/dev/null)"
[ -f smirk/assets/face_landmarker.task ] || curl -sL -o smirk/assets/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
echo FETCH_DONE
