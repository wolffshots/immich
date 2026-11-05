#!/bin/bash
# Pushes the sm_52 image to ghcr.io from inside the Podman machine. Log in first:
#   gh auth token | podman machine ssh "sudo podman login ghcr.io -u wolffshots --password-stdin"
# Then run it as a systemd unit:
#   podman machine ssh "sudo systemd-run --unit=sm52-push --collect /bin/bash /mnt/c/<path-to-this-folder>/push.sh"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMMICH_VERSION="${IMMICH_VERSION:-v3.3.6}"
SRC="localhost/immich-machine-learning:${IMMICH_VERSION}-cuda-sm52"
DST="ghcr.io/wolffshots/immich-machine-learning:${IMMICH_VERSION}-cuda-sm52"
L="$DIR/push.log"
{ podman tag "$SRC" "$DST" && podman push --format v2s2 "$DST"; } > "$L" 2>&1
echo "exit=$?" >> "$L"
