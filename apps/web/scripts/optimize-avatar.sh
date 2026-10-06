#!/usr/bin/env bash
# Rebuilds public/avatars/interviewer.glb from the pinned CC0 source. Requires network and npx.
set -euo pipefail
SRC_URL="https://raw.githubusercontent.com/met4citizen/TalkingHead/b3e277b3b46f88e557bf28a2c5612a5b04e075c3/avatars/mpfb.glb"
SRC_SHA="63c645a2a863b9972e9a9c2ed576a1de4c390b8475508e1473e69c87a3ee299c"
T=$(mktemp -d)
curl -sSL -o "$T/src.glb" "$SRC_URL"
echo "$SRC_SHA  $T/src.glb" | sha256sum -c -
G="npx -y @gltf-transform/cli@4"
$G dedup "$T/src.glb" "$T/1.glb"
$G prune "$T/1.glb" "$T/2.glb"
$G quantize "$T/2.glb" "$T/3.glb"
$G resize "$T/3.glb" "$T/4.glb" --width 2048 --height 2048
$G webp "$T/4.glb" "$T/5.glb" --quality 88
$G meshopt "$T/5.glb" "$(dirname "$0")/../public/avatars/interviewer.glb" --level high
echo "wrote public/avatars/interviewer.glb"
