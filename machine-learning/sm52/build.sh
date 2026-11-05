#!/bin/bash
# Builds the sm_52 Immich ML image inside the Podman machine. Run it as a systemd unit:
#   podman machine ssh "sudo systemd-run --unit=sm52-build --collect /bin/bash /mnt/c/<path-to-this-folder>/build.sh"
# Override the versions with: sudo systemd-run ... --setenv=IMMICH_VERSION=v3.4.0 --setenv=ORT_VERSION=v1.27.1 ...
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMMICH_VERSION="${IMMICH_VERSION:-v3.3.6}"
ORT_VERSION="${ORT_VERSION:-v1.27.1}"
PARALLEL="${PARALLEL:-3}"
L="$DIR/build.log"
podman build \
  --format docker \
  --build-arg IMMICH_VERSION="$IMMICH_VERSION" \
  --build-arg ORT_VERSION="$ORT_VERSION" \
  --build-arg PARALLEL="$PARALLEL" \
  -t "localhost/immich-machine-learning:${IMMICH_VERSION}-cuda-sm52" \
  "$DIR" > "$L" 2>&1
echo "exit=$?" >> "$L"
