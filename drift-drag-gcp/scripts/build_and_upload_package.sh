#!/usr/bin/env bash
# STEP 7 (after Kafka/Redis VMs exist, before creating the Dataproc
# cluster). Builds/collects everything Dataproc and Airflow need from GCS
# and uploads it: the driftdrag Python package (zipped, for --py-files),
# the Scala shard-partitioner fat jar (built via sbt, reused unmodified
# from drift-drag/), a copy of spark_embed_job.py (Airflow's Dataproc
# operator needs a GCS URI, unlike the submit_*.sh scripts which can
# auto-upload a local path), and the Dataproc initialization action.
# Re-run this any time the driftdrag package or the Scala job changes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRIFTDRAG_DIR="$REPO_ROOT/drift-drag"
BUILD_DIR="$SCRIPT_DIR/../build"
mkdir -p "$BUILD_DIR"

echo "== 1/4: zipping driftdrag package =="
# Uses Python's stdlib zipfile instead of the `zip` CLI -- not guaranteed to
# be installed (e.g. plain Git Bash on Windows doesn't ship it), whereas
# python is already a hard requirement everywhere else in this project.
python3 - "$DRIFTDRAG_DIR/src" "$BUILD_DIR/driftdrag.zip" <<'PYEOF'
import sys, zipfile
from pathlib import Path

src_dir, out_zip = Path(sys.argv[1]), Path(sys.argv[2])
pkg_dir = src_dir / "driftdrag"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in pkg_dir.rglob("*"):
        if "__pycache__" in path.parts or path.is_dir():
            continue
        zf.write(path, path.relative_to(src_dir))
print(f"wrote {out_zip}")
PYEOF

echo "== 2/4: building Scala shard-partitioner fat jar (sbt assembly) =="
SBT_BIN="$DRIFTDRAG_DIR/.tools/sbt/bin/sbt.bat"
JAVA_HOME_FOR_SBT="$DRIFTDRAG_DIR/.tools/jdk-17.0.19+10"
JAR_PATH="$DRIFTDRAG_DIR/scala/shard-partitioner/target/scala-2.12/shard-partitioner-assembly-0.1.0.jar"
if [[ ! -f "$JAR_PATH" ]]; then
  if [[ -f "$SBT_BIN" ]]; then
    echo "Building with the portable sbt already set up under drift-drag/.tools/..."
    ( cd "$DRIFTDRAG_DIR/scala/shard-partitioner" && JAVA_HOME="$JAVA_HOME_FOR_SBT" "$SBT_BIN" -no-colors assembly )
  else
    echo "ERROR: $JAR_PATH not found and no sbt available at $SBT_BIN."
    echo "Build it once with: cd ${DRIFTDRAG_DIR}/scala/shard-partitioner && sbt assembly"
    exit 1
  fi
fi
cp "$JAR_PATH" "$BUILD_DIR/shard-partitioner-assembly-0.1.0.jar"

echo "== 3/4: copying spark_embed_job.py + init_actions.sh =="
cp "$DRIFTDRAG_DIR/src/driftdrag/ingest/spark_embed_job.py" "$BUILD_DIR/spark_embed_job.py"
cp "$SCRIPT_DIR/../05_dataproc/init_actions.sh" "$BUILD_DIR/init_actions.sh"

echo "== 4/4: uploading to gs://${GCS_BUCKET}/code/ =="
gcloud storage cp "$BUILD_DIR/driftdrag.zip" "gs://${GCS_BUCKET}/code/driftdrag.zip"
gcloud storage cp "$BUILD_DIR/shard-partitioner-assembly-0.1.0.jar" "gs://${GCS_BUCKET}/code/shard-partitioner-assembly-0.1.0.jar"
gcloud storage cp "$BUILD_DIR/spark_embed_job.py" "gs://${GCS_BUCKET}/code/spark_embed_job.py"
gcloud storage cp "$BUILD_DIR/init_actions.sh" "gs://${GCS_BUCKET}/code/init_actions.sh"

echo ""
echo "Done. Contents of gs://${GCS_BUCKET}/code/:"
gcloud storage ls "gs://${GCS_BUCKET}/code/"
