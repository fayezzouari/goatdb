#!/usr/bin/env bash
# Fetch ann-benchmarks datasets into ./data (gitignored).
#
#   ./download.sh                         # both datasets
#   ./download.sh sift-128-euclidean      # one
#
# Sizes: sift-128-euclidean.hdf5 ~ 501 MB (1M x 128 train, 10k test)
#        glove-100-angular.hdf5  ~ 463 MB (1.18M x 100 train, 10k test)
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data

names=("$@")
[ ${#names[@]} -eq 0 ] && names=(sift-128-euclidean glove-100-angular)

for name in "${names[@]}"; do
  case "$name" in
    sift-128-euclidean | glove-100-angular) ;;
    *) echo "unknown dataset: $name" >&2; exit 1 ;;
  esac
  out="data/$name.hdf5"
  if [ -s "$out" ]; then
    echo "$out already present ($(du -h "$out" | cut -f1))"
    continue
  fi
  echo "downloading $name ..."
  curl -fL --retry 3 -C - -o "$out.part" "http://ann-benchmarks.com/$name.hdf5"
  mv "$out.part" "$out"
  echo "$out: $(du -h "$out" | cut -f1)"
done
