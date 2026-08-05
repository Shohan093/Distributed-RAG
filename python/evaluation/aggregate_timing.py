import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)


def main():
    records = []
    with open("results/tables/timing_log.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    # in aggregate_timing.py, after loading records:
    WARMUP_REQUESTS_TO_EXCLUDE = 2
    records = records[WARMUP_REQUESTS_TO_EXCLUDE:]
    print(f"Excluded {WARMUP_REQUESTS_TO_EXCLUDE} warm-up requests, aggregating remaining {len(records)}")

    breakdown = {
        "encode": [r["encode_ms"] for r in records],
        "shard_fanout": [r["shard_fanout_ms"] for r in records],
        "fuse": [r["fuse_ms"] for r in records],
    }

    with open("results/tables/timing_breakdown.json", "w") as f:
        json.dump(breakdown, f)

    print(f"Aggregated {len(records)} requests into results/tables/timing_breakdown.json")


if __name__ == "__main__":
    main()