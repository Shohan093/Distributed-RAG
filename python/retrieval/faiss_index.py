import numpy as np
import faiss


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build_flat_index(embeddings, doc_ids):
    embeddings = normalize(embeddings).astype("float32")
    dim = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, np.array(doc_ids, dtype="int64"))
    return index


def build_ivf_index(embeddings, doc_ids, nlist=1333, nprobe=16):
    embeddings = normalize(embeddings).astype("float32")
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    base_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    base_index.train(embeddings)
    base_index.nprobe = nprobe
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, np.array(doc_ids, dtype="int64"))
    return index