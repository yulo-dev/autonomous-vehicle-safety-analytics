#!/usr/bin/env bash
# Download a small WOMD subset (validation shards) into data/raw.
#
# Prerequisites:
#   1. Accept the Waymo Open Dataset license at https://waymo.com/open with a Google account.
#   2. Install the Google Cloud CLI and run `gcloud auth login` with that same account.
#
# Usage:
#   ./scripts/download_subset.sh            # shard 0 only
#   N_SHARDS=3 ./scripts/download_subset.sh
#   WOMD_BUCKET=gs://<bucket-from-waymo-download-page> ./scripts/download_subset.sh
#
# The bucket name encodes the dataset version. Confirm it on the Waymo download
# page and override WOMD_BUCKET if it differs from the default below.
set -euo pipefail

WOMD_BUCKET="${WOMD_BUCKET:-gs://waymo_open_dataset_motion_v_1_3_1}"
SPLIT="${SPLIT:-validation}"
N_SHARDS="${N_SHARDS:-1}"
TOTAL="${TOTAL_SHARDS:-00150}"
DEST="${DEST:-data/raw}"

command -v gcloud >/dev/null || { echo "gcloud not found; install the Google Cloud CLI" >&2; exit 1; }
mkdir -p "$DEST"

for ((i = 0; i < N_SHARDS; i++)); do
  shard=$(printf "%s.tfrecord-%05d-of-%s" "$SPLIT" "$i" "$TOTAL")
  src="$WOMD_BUCKET/uncompressed/scenario/$SPLIT/$shard"
  if [[ -f "$DEST/$shard" ]]; then
    echo "skip $shard (exists)"
    continue
  fi
  echo "fetch $src"
  gcloud storage cp "$src" "$DEST/"
done
