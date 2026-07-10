"""Orchestrates one full Stage-0-through-Stage-4 pipeline run:

  create Dataproc cluster -> submit Scala shard-partitioner -> submit
  PySpark embed job -> publish HDFS embeddings to GCS -> delete Dataproc
  cluster (always, even on failure -- this is the cost-safety net) ->
  run Ray drift-detection -> run Ray compensation -> run Ray scheduler.

This is the Airflow half of PLAN.md's Stage 0-4 pipeline: Dataproc/Ray/
Kafka/Redis do the actual work (see 05_dataproc/ and 06_ray_cluster/ for
the same operations run by hand); this DAG exists so a full run is one
button in the Airflow UI instead of nine manual script invocations, and so
the Dataproc-delete step is *guaranteed* to run even if an earlier task
fails -- the single most important cost-safety property in this whole
project (see COST_BUDGET.md).

Ray steps use BashOperator + `gcloud compute ssh` (same pattern as
../../06_ray_cluster/run_ray_job.sh) rather than SSHOperator, so there is
no separate Airflow SSH Connection to configure by hand -- auth flows
through this VM's attached service account exactly like the Dataproc
operators do.

Constants below MUST match ../../env.sh -- there is no shared source
between the two because Airflow DAGs run in a separate Python process on
the Airflow VM, not alongside the bash scripts.
"""

from __future__ import annotations

import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitJobOperator,
)
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ID = "drift-rag"
REGION = "asia-south1"
ZONE = "asia-south1-a"
GCS_BUCKET = "drift-rag-driftdrag-lake"
CLUSTER_NAME = "driftdrag-cluster"
KAFKA_VM = "driftdrag-kafka"
REDIS_VM = "driftdrag-redis"
RAY_HEAD_VM = "driftdrag-ray-head"
NUM_SHARDS = "32"

CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n2-standard-4",
        "disk_config": {"boot_disk_size_gb": 50},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "n2-standard-4",
        "disk_config": {"boot_disk_size_gb": 50},
    },
    "initialization_actions": [{"executable_file": f"gs://{GCS_BUCKET}/code/init_actions.sh"}],
    "software_config": {"image_version": "2.2-debian12"},
}

SPARK_KAFKA_PROPS = {"spark.jars.packages": "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"}


def _describe_ip(vm_name: str) -> str:
    return (
        f"gcloud compute instances describe {vm_name} --zone {ZONE} --project {PROJECT_ID} "
        "--format='value(networkInterfaces[0].networkIP)'"
    )


def _ray_ssh_command(job_file: str, extra_env: str = "") -> str:
    """Mirrors 06_ray_cluster/run_ray_job.sh exactly -- keep the two in sync
    if either changes.
    """
    redis_ip_cmd = _describe_ip(REDIS_VM)
    return (
        f'REDIS_IP=$({redis_ip_cmd}) && '
        f'gcloud compute ssh {RAY_HEAD_VM} --zone {ZONE} --project {PROJECT_ID} --command '
        f'"cd ~/driftdrag_pkg && GCS_BUCKET={GCS_BUCKET} REDIS_HOST=$REDIS_IP {extra_env} '
        f'/opt/driftdrag-venv/bin/python ray_jobs/{job_file}"'
    )


default_args = {
    "owner": "driftdrag",
    "retries": 0,
}

with DAG(
    dag_id="drift_drag_pipeline",
    description="Stage 0 (Dataproc: Scala+PySpark) -> Stage 2-4 (Ray) -- one full run",
    schedule=None,  # trigger manually from the Airflow UI; not a recurring cron DAG
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["drift-drag"],
) as dag:

    get_kafka_ip = BashOperator(task_id="get_kafka_ip", bash_command=_describe_ip(KAFKA_VM))

    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=CLUSTER_NAME,
        cluster_config=CLUSTER_CONFIG,
    )

    submit_shard_partitioner = DataprocSubmitJobOperator(
        task_id="submit_shard_partitioner",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "reference": {"project_id": PROJECT_ID},
            "placement": {"cluster_name": CLUSTER_NAME},
            "spark_job": {
                "jar_file_uris": [f"gs://{GCS_BUCKET}/code/shard-partitioner-assembly-0.1.0.jar"],
                "main_class": "com.driftdrag.ingest.ShardPartitionerJob",
                "properties": SPARK_KAFKA_PROPS,
                "args": [
                    "--source=kafka",
                    "--bootstrap-servers={{ ti.xcom_pull(task_ids='get_kafka_ip') }}:9092",
                    "--topic=docs-raw",
                    f"--output-path=gs://{GCS_BUCKET}/shard_assignments",
                    f"--checkpoint-path=gs://{GCS_BUCKET}/_checkpoints/shard_partitioner",
                    f"--num-shards={NUM_SHARDS}",
                    "--model-version=v0",
                    "--trigger-once=true",
                ],
            },
        },
    )

    submit_embed_job = DataprocSubmitJobOperator(
        task_id="submit_embed_job",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "reference": {"project_id": PROJECT_ID},
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"gs://{GCS_BUCKET}/code/spark_embed_job.py",
                "python_file_uris": [f"gs://{GCS_BUCKET}/code/driftdrag.zip"],
                "properties": SPARK_KAFKA_PROPS,
                "args": [
                    "--source=kafka",
                    "--bootstrap-servers={{ ti.xcom_pull(task_ids='get_kafka_ip') }}:9092",
                    "--topic=docs-raw",
                    "--output-path=hdfs:///driftdrag/embeddings",
                    f"--checkpoint-path=gs://{GCS_BUCKET}/_checkpoints/spark_embed_job",
                    f"--num-shards={NUM_SHARDS}",
                    "--model-version=v0",
                    "--backend-name=hashing",
                ],
            },
        },
    )

    publish_to_gcs = DataprocSubmitJobOperator(
        task_id="publish_hdfs_to_gcs",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "reference": {"project_id": PROJECT_ID},
            "placement": {"cluster_name": CLUSTER_NAME},
            "hadoop_job": {
                "main_class": "org.apache.hadoop.tools.DistCp",
                "args": ["hdfs:///driftdrag/embeddings", f"gs://{GCS_BUCKET}/embeddings"],
            },
        },
    )

    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=CLUSTER_NAME,
        trigger_rule=TriggerRule.ALL_DONE,  # run even if an earlier task failed -- cost safety net
    )

    run_drift_detection = BashOperator(
        task_id="run_ray_drift_detection",
        bash_command=_ray_ssh_command("parallel_drift_detection.py"),
    )

    run_compensation = BashOperator(
        task_id="run_ray_compensation",
        bash_command=_ray_ssh_command("parallel_compensation.py"),
    )

    run_scheduler = BashOperator(
        task_id="run_ray_scheduler",
        bash_command=_ray_ssh_command("run_scheduler.py", extra_env="BUDGET_GPU_HOURS=2.0"),
    )

    (
        get_kafka_ip
        >> create_cluster
        >> submit_shard_partitioner
        >> submit_embed_job
        >> publish_to_gcs
        >> delete_cluster
        >> run_drift_detection
        >> run_compensation
        >> run_scheduler
    )
