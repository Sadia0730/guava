#!/usr/bin/env bash
# Bring an extracted laptop package (guava_live_20261103) up to date from this git checkout: copies the live app, the background
# worker with the live appearance update, the ArcFace module and the metrics, then fetches the ArcFace model the update needs
# (insightface antelopev2 glintr100.onnx, research / non-commercial use; same bytes as the server copy, checked by SHA-256).
#   cd <your git clone> && git pull && bash phase3_dnt/live/update_package.sh /data/GUAVA/guava_live
set -euo pipefail
PKG=$(realpath "${1:?usage: update_package.sh PACKAGE_DIR}")
cd "$(dirname "$0")/../.."
[ -f "$PKG/MANIFEST.json" ] || { echo "$PKG is not an extracted guava_live package (no MANIFEST.json)"; exit 1; }
mkdir -p "$PKG/phase3_dnt/live" "$PKG/phase3_dnt/worker6" "$PKG/main"
cp phase3_dnt/live/*.py phase3_dnt/live/*.sh phase3_dnt/live/*.md phase3_dnt/live/*.txt "$PKG/phase3_dnt/live/"
cp phase3_dnt/live/README_LIVE.md "$PKG/README_LIVE.md"
cp phase3_dnt/worker6/*.py "$PKG/phase3_dnt/worker6/"
cp phase3_dnt/onnx_arcface_torch.py "$PKG/phase3_dnt/"
cp main/__init__.py main/metrics.py "$PKG/main/"
ARC="$PKG/live_assets/arcface/glintr100.onnx"; SHA=4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf
URL=https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/face/models/antelopev2/glintr100.onnx
if ! echo "$SHA  $ARC" | sha256sum -c --status 2>/dev/null; then
  mkdir -p "$(dirname "$ARC")"; echo "downloading the ArcFace model (261 MB) ..."
  if command -v wget >/dev/null; then wget -q --show-progress -O "$ARC" "$URL"; else curl -L -o "$ARC" "$URL"; fi
  echo "$SHA  $ARC" | sha256sum -c
fi
echo "package $PKG updated to $(git rev-parse --short HEAD)"
