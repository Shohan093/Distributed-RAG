import json
import os
import numpy as np

def checkpointed_encode(model, texts, doc_ids, output_dir, batch_size=2000):
    """Encodes texts in resumable batches. Returns nothing; writes final
    doc_embeddings.npy and doc_ids.json to output_dir once complete."""
    final_embeddings_path = os.path.join(output_dir, "doc_embeddings.npy")
    final_ids_path = os.path.join(output_dir, "doc_ids.json")

    if os.path.exists(final_embeddings_path) and os.path.exists(final_ids_path):
        print(f"{output_dir}: already complete, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")

    n_batches = (len(texts) + batch_size - 1) // batch_size

    completed_batches = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            completed_batches = set(json.load(f))
        print(f"Resuming: {len(completed_batches)}/{n_batches} batches already done.")

    for batch_idx in range(n_batches):
        if batch_idx in completed_batches:
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        print(f"Batch {batch_idx + 1}/{n_batches} ({start}:{end})...")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

        batch_path = os.path.join(checkpoint_dir, f"batch_{batch_idx:05d}.npy")
        np.save(batch_path, batch_embeddings)

        completed_batches.add(batch_idx)
        with open(manifest_path, "w") as f:
            json.dump(sorted(completed_batches), f)

    print("All batches complete. Assembling final file...")
    all_embeddings = [np.load(os.path.join(checkpoint_dir, f"batch_{i:05d}.npy")) for i in range(n_batches)]
    final_embeddings = np.vstack(all_embeddings)
    np.save(final_embeddings_path, final_embeddings)
    with open(final_ids_path, "w") as f:
        json.dump(doc_ids, f)

    print(f"Saved {final_embeddings.shape} to {final_embeddings_path}")

    for i in range(n_batches):
        os.remove(os.path.join(checkpoint_dir, f"batch_{i:05d}.npy"))
    os.remove(manifest_path)
    os.rmdir(checkpoint_dir)