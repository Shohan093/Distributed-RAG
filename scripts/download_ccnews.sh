#!/bin/bash
# scripts/download_ccnews.sh
set -e

OUTPUT_DIR="data/raw"
mkdir -p "$OUTPUT_DIR"

BASE_URL="https://huggingface.co/datasets/vblagoje/cc_news/resolve/refs%2Fconvert%2Fparquet/plain_text/train"

for i in 0000 0001 0002 0003 0004; do
  OUT_FILE="$OUTPUT_DIR/cc_news-train-${i}.parquet"
  if [ -f "$OUT_FILE" ]; then
    echo "Shard $i already exists, skipping."
    continue
  fi
  echo "Downloading shard $i..."
  curl --http1.1 -fL --retry 3 --retry-delay 5 -o "$OUT_FILE" "$BASE_URL/${i}.parquet"
done

echo "Done. Files:"
ls -lh "$OUTPUT_DIR"
