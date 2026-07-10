#!/usr/bin/env bash
# Single source of truth for every script in this directory. Every other
# script does `source "$(dirname "$0")/../env.sh"` (path depth varies) and
# never hardcodes a project/region/machine size itself -- change values
# here once, everything downstream picks it up.
set -euo pipefail

export PROJECT_ID="drift-rag"
export REGION="asia-south1"
export ZONE="asia-south1-a"

# GCS bucket names are GLOBAL across every GCP customer, not just your
# project -- if this exact name is taken, pick another (e.g. append your
# initials) and re-export before running anything under 02_storage/.
export GCS_BUCKET="${PROJECT_ID}-driftdrag-lake"

# -- VM names --
export KAFKA_VM="driftdrag-kafka"
export REDIS_VM="driftdrag-redis"
export AIRFLOW_VM="driftdrag-airflow"
export RAY_HEAD_VM="driftdrag-ray-head"
export GPU_VM="driftdrag-gpu"

# -- machine sizing (deliberately small -- see COST_BUDGET.md) --
export SMALL_MACHINE_TYPE="e2-medium"     # Kafka, Redis, Airflow: 2 vCPU / 4GB, a few cents/hour
export RAY_MACHINE_TYPE="e2-standard-4"   # Ray head: 4 vCPU / 16GB, needed for parallel per-shard work
export GPU_MACHINE_TYPE="n1-standard-4"   # paired with a T4 (quota=1 in this project)
export GPU_ACCELERATOR="type=nvidia-tesla-t4,count=1"
# Verified current (gcloud compute images list --project deeplearning-platform-release
# --filter="family~pytorch") -- re-check this if it's been a while, DLVM
# family names change as CUDA/PyTorch versions are refreshed.
export GPU_IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
export GPU_IMAGE_PROJECT="deeplearning-platform-release"

# -- Dataproc (ephemeral -- created per job run, deleted right after) --
export DATAPROC_CLUSTER="driftdrag-cluster"
export DATAPROC_IMAGE_VERSION="2.2-debian12"   # Spark 3.5.x / Hadoop 3.3.x / HDFS+YARN included
export DATAPROC_MASTER_MACHINE="n2-standard-4"
export DATAPROC_WORKER_MACHINE="n2-standard-4"
export DATAPROC_NUM_WORKERS="2"

# -- shared network --
export NETWORK="default"     # see 01_network/ for why "default" instead of a custom VPC

# -- app-level config (mirrors drift-drag/.env.example, GCP-specific values) --
export NUM_SHARDS="32"
export EMBEDDING_DIM="128"
export KAFKA_TOPIC_DOCS="docs-raw"
export KAFKA_TOPIC_LOGS="logs-raw"
