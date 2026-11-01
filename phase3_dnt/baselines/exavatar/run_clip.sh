#!/bin/bash
# ExAvatar baseline on one AvatarTest-v1 clip: preprocessing (ExAvatar's own face pipeline, body = GT) -> avatar training on
# the recon frames -> test renders (GT body + ExAvatar face fit, black bg) -> E/renders/exavatar/<dataset>/<clip>/.
# usage: run_clip.sh <dataset> <clip> [gpu|auto] [--keep]
# Everything heavy stays in /dev/shm; only renders + run.json + small logs are copied to E (SMB share, plain copies).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
DS=$1; CLIP=$2; GPU=${3:-auto}; KEEP=${4:-}
E=/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1
SID=${DS}__${CLIP}
CLIPDIR=$EXA_ROOT/inputs/$DS/$CLIP
WORK=$EXA_ROOT/work/$SID
LOGD=$EXA_ROOT/runlogs/$SID
FIT=$EXA_SRC/fitting
AV=$EXA_SRC/avatar
OUT=$E/renders/exavatar/$DS/$CLIP
mkdir -p "$LOGD"
TIMES=$LOGD/times.txt; : > "$TIMES"
stamp() { date +%s.%N; }
mark() { echo "$1 $(stamp)" >> "$TIMES"; echo "[$(date '+%F %T')] $1"; }

GPU_ORDER=${GPU_ORDER:-"3 2 4 5 6 7"}   # coordinator: GPUs 0 (GART) and 1 (GUAVA rows) are taken
pick_gpu() {  # first GPU of GPU_ORDER with < 2 GB in use
  local used
  for g in $GPU_ORDER; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
    if [ "$used" -lt 2048 ]; then echo $g; return 0; fi
  done
  return 1
}
if [ "$GPU" = auto ]; then  # wait (up to 12 h) until a GPU with < 2 GB in use is available
  for i in $(seq 1 2880); do GPU=$(pick_gpu) && break; GPU=; sleep 15; done
  [ -n "$GPU" ] || { echo "no free GPU"; exit 3; }
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
[ "$used" -lt 2048 ] || { echo "GPU $GPU has $used MiB in use"; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU
# 8 CPU threads per process (shared 224-core host; ExAvatar's CPU-side ops are ~2x faster than with the default
# all-core thread pools); ExAvatar's tools call `python` from PATH (as after `conda activate`)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PATH_MAIN=$EXA_ENV/bin:$PATH; PATH_POSE=$EXA_ENV_POSE/bin:$PATH; export PATH=$PATH_MAIN
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)
echo "sid=$SID gpu=$GPU ($GPUNAME)" | tee "$LOGD/gpu.txt"

# -1. stage the input package in /dev/shm (I/O of our infrastructure; timed separately, not part of train_s)
mark copy_start
mkdir -p "$CLIPDIR"
[ -f "$CLIPDIR/frames.json" ] || cp -r "$E/base/$DS/$CLIP/." "$CLIPDIR/"
mark copy_end

mark train_start   # == start of the method's own wall time (preprocessing + face fitting + training)
# 0. ExAvatar subject folders + GT conversion (CPU)
$EXA_ENV/bin/python "$HERE/prep_clip.py" --clip_dir "$CLIPDIR" --sid "$SID" > "$LOGD/00_prep.log" 2>&1
mark prep_end
# 1. DECA (initial FLAME), ExAvatar's tools/DECA/run_deca.py, on recon frames and on test frames (separate folders)
for SPLIT in recon test; do
  ( export PATH=$PATH_MAIN; cd $FIT/tools/DECA && flock $EXA_ROOT/.lock_deca $EXA_ENV/bin/python run_deca.py --root_path $WORK/fit_$SPLIT ) > "$LOGD/01_deca_$SPLIT.log" 2>&1
  [ -f $WORK/fit_$SPLIT/flame_init/shape_param.json ] || { echo "DECA failed ($SPLIT)"; exit 1; }
done
mark deca_end
# 2. mmpose whole-body 2D keypoints, ExAvatar's tools/mmpose/run_mmpose.py
for SPLIT in recon test; do
  ( export PATH=$PATH_POSE; cd $FIT/tools/mmpose && flock $EXA_ROOT/.lock_mmpose $EXA_ENV_POSE/bin/python run_mmpose.py --root_path $WORK/fit_$SPLIT ) > "$LOGD/02_mmpose_$SPLIT.log" 2>&1
  n=$(ls $WORK/fit_$SPLIT/keypoints_whole_body | wc -l); m=$(ls $WORK/fit_$SPLIT/frames | wc -l)
  [ "$n" = "$m" ] || { echo "mmpose: $n keypoint files for $m frames ($SPLIT)"; exit 1; }
done
mark mmpose_end
# 3. ExAvatar face fitting on the recon frames (body = GT), then ExAvatar's own texture unwrap (main/unwrap.py)
( cd $FIT/main && $EXA_ENV/bin/python "$HERE/fit_face_gt.py" --subject_id ${SID}_recon --gt_conv $WORK/gt_conversion.json --stage recon ) > "$LOGD/03_fit_recon.log" 2>&1
rm -rf $WORK/fit_recon/smplx_optimized; mv $FIT/output/result/${SID}_recon/* $WORK/fit_recon/
( cd $FIT/main && $EXA_ENV/bin/python unwrap.py --subject_id ${SID}_recon ) > "$LOGD/04_unwrap.log" 2>&1
mv $FIT/output/result/${SID}_recon/unwrapped_textures/* $WORK/fit_recon/smplx_optimized/
mark fit_recon_end
# 4. ExAvatar face fitting on the test frames (face only; ID frozen at the recon result)
( cd $FIT/main && $EXA_ENV/bin/python "$HERE/fit_face_gt.py" --subject_id ${SID}_test --gt_conv $WORK/gt_conversion.json --stage test --id_dir $WORK/fit_recon/smplx_optimized ) > "$LOGD/05_fit_test.log" 2>&1
rm -rf $WORK/fit_test/smplx_optimized; mv $FIT/output/result/${SID}_test/* $WORK/fit_test/
mark fit_test_end
# 5. assemble avatar folder: GT body + smoothed face params (ExAvatar smoother), ID files, keypoints, meshes
( cd $FIT/tools && $EXA_ENV/bin/python "$HERE/finalize_params.py" --clip_dir "$CLIPDIR" --sid "$SID" ) > "$LOGD/06_finalize.log" 2>&1
# 6. background point cloud (ExAvatar's static-background route, tools/Depth-Anything-V2/run_depth_anything.py)
( export PATH=$PATH_MAIN; cd $FIT/tools/Depth-Anything-V2 && $EXA_ENV/bin/python run_depth_anything.py --root_path $WORK/avatar ) > "$LOGD/07_depth.log" 2>&1
[ -f $WORK/avatar/bkg_point_cloud.txt ] || { echo "no bkg point cloud"; exit 1; }
mark preprocess_end
# 7. avatar training (ExAvatar main/train.py, default config, dataset = 'Custom')
( cd $AV/main && $EXA_ENV/bin/python train.py --subject_id $SID ) > "$LOGD/08_train.log" 2>&1
[ -f $AV/output/model_dump/$SID/snapshot_4.pth ] || { echo "training failed"; exit 1; }
mark train_end
# 8. test renders
( export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; cd $AV/main && $EXA_ENV/bin/python "$HERE/render_test.py" --clip_dir "$CLIPDIR" --sid $SID --out_dir $WORK/renders ) > "$LOGD/09_render.log" 2>&1
mark render_end

# run.json + copy to E
grep -E "Epoch 4/5 itr" "$LOGD/08_train.log" | tail -1 > "$LOGD/train_last_itr.txt" || true
$EXA_ENV/bin/python "$HERE/write_run_json.py" --sid $SID --clip_dir "$CLIPDIR" --gpu_name "$GPUNAME" --gpu_index $GPU --logd "$LOGD" --out "$WORK/run.json"
mkdir -p "$OUT"
cp $WORK/renders/*.png "$OUT/"
cp $WORK/run.json "$OUT/run.json"
mkdir -p "$OUT/logs"
for f in times.txt gpu.txt 00_prep.log 06_finalize.log 09_render.log; do cp "$LOGD/$f" "$OUT/logs/"; done
grep -v "^\[recon\]\|^\[test\]" "$LOGD/03_fit_recon.log" | tail -n 20 > "$OUT/logs/03_fit_recon_tail.log" || true
tail -n 5 "$LOGD/08_train.log" > "$OUT/logs/08_train_tail.log"
cp $WORK/finalize_stats.json $WORK/render_stats.json $WORK/gt_conversion.json "$OUT/logs/" 2>/dev/null || true
n_out=$(ls "$OUT"/*.png | wc -l); n_test=$(python3 -c "import json;print(len(json.load(open('$CLIPDIR/frames.json'))['test']))")
echo "renders written: $n_out / $n_test -> $OUT"

# cleanup: checkpoints / caches / intermediate files stay only in /dev/shm and are deleted after the renders are written
if [ "$KEEP" != "--keep" ]; then
  rm -rf $AV/output/model_dump/$SID $AV/output/result/$SID $FIT/output/result/${SID}_recon $FIT/output/result/${SID}_test
  rm -f $AV/data/Custom/data/$SID $FIT/data/Custom/data/${SID}_recon $FIT/data/Custom/data/${SID}_test
  mkdir -p $EXA_ROOT/runlogs_keep/$SID && cp $WORK/run.json $WORK/*.json $EXA_ROOT/runlogs_keep/$SID/ 2>/dev/null || true
  rm -rf $WORK
fi
mark done
