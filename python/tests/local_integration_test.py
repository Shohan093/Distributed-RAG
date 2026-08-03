import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

from retrieval.router import Router
from retrieval.shard import Shard

# A real query -- borrowed from one of our eval set's actual article titles,
# so we know it SHOULD have a genuine, findable match in its own epoch's index.
QUERY_TEXT = "North Korea suspected behind murder of leader's half-brother"

def main():
    print("Loading router (v3 encoder)...")
    router = Router()

    print("Loading all three shards...")
    shards = {
        1: Shard(epoch=1, index_type="flat"),
        2: Shard(epoch=2, index_type="flat"),
        3: Shard(epoch=3, index_type="flat"),
    }

    print(f"\nQuery: {QUERY_TEXT}\n")
    query_emb = router.encode_query(QUERY_TEXT)

    results_per_shard = []
    for epoch, shard in shards.items():
        results = shard.search(query_emb, k=5)
        print(f"--- Shard {epoch} (epoch {epoch}) top results ---")
        for r in results:
            print(f"  {r['doc_id']}  score={r['score']:.4f}  {r['title'][:60]}")
        results_per_shard.append(results)

    fused = router.fuse_results(results_per_shard, k=10)

    print("\n=== Fused top-10 across all shards ===")
    for r in fused:
        print(f"  doc={r['doc_id']}  fused_score={r['fused_score']:.4f}  {r['title'][:60]}")

if __name__ == "__main__":
    main()