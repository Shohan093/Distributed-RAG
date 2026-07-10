"""Pluggable embedding backends (Stage 0).

Real experiments (E0-E7) embed with nomic-embed-text-v1.5 / BGE-base-en-v1.5,
continually fine-tuned to produce the version sequence v1..v5 (QDC recipe).
Downloading and fine-tuning those models is out of scope for this scaffold's
smoke test (no network/GPU guarantee), so ``HashingDriftBackend`` below is a
dependency-free stand-in that simulates a *known-ground-truth* drift sequence
-- each version v_i relates to v_{i-1} by a fixed small rotation, so
detectors and compensation operators can be unit-tested against an exact
answer before ever touching a real model. Swap in
``SentenceTransformerBackend`` (or a HF ``AutoModel`` wrapper) for real runs;
both implement the same two-method interface so no caller code changes.
"""

from __future__ import annotations

import hashlib

import numpy as np


class EmbeddingBackend:
    dim: int

    def embed(self, texts: list[str], model_version: str) -> np.ndarray:
        raise NotImplementedError


class HashingDriftBackend(EmbeddingBackend):
    def __init__(self, dim: int = 128, base_seed: int = 0, drift_angle: float = 0.05):
        self.dim = dim
        self.base_seed = base_seed
        self.drift_angle = drift_angle
        self._rotation_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _version_index(model_version: str) -> int:
        digits = "".join(ch for ch in model_version if ch.isdigit())
        return int(digits) if digits else 0

    def _rotation_for_version(self, version_idx: int) -> np.ndarray:
        if version_idx in self._rotation_cache:
            return self._rotation_cache[version_idx]
        r = np.eye(self.dim)
        for i in range(version_idx):
            rng = np.random.default_rng(self.base_seed + i + 1)
            a, b = rng.choice(self.dim, size=2, replace=False)
            step = np.eye(self.dim)
            step[a, a] = np.cos(self.drift_angle)
            step[b, b] = np.cos(self.drift_angle)
            step[a, b] = -np.sin(self.drift_angle)
            step[b, a] = np.sin(self.drift_angle)
            r = step @ r
        self._rotation_cache[version_idx] = r
        return r

    def _stable_seed(self, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")

    def _base_embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(self._stable_seed(text))
            vec = rng.normal(size=self.dim)
            out[i] = vec / (np.linalg.norm(vec) + 1e-12)
        return out

    def embed(self, texts: list[str], model_version: str = "v0") -> np.ndarray:
        base = self._base_embed(texts)
        rotation = self._rotation_for_version(self._version_index(model_version))
        return (base @ rotation.T).astype(np.float32)


class SentenceTransformerBackend(EmbeddingBackend):
    """Thin wrapper so real backbones (Section 5: nomic-embed-text-v1.5,
    BGE-base-en-v1.5) drop in without touching pipeline code. Each fine-tuned
    checkpoint on disk/registry is one "model_version"; pass a
    ``{model_version: checkpoint_path_or_name}`` map.
    """

    def __init__(
        self,
        checkpoints: dict[str, str],
        dim: int,
        trust_remote_code: bool = False,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self.checkpoints = checkpoints
        self.dim = dim
        self.trust_remote_code = trust_remote_code
        # nomic-embed-text-v1.5 (and e5/BGE-style models) are asymmetric:
        # queries and documents need different instruction prefixes or
        # retrieval quality collapses -- this is not optional formatting,
        # it's how the model was trained. Leave both "" for symmetric
        # backbones that don't use prefixes.
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._loaded: dict[str, object] = {}

    def _get_model(self, model_version: str):
        if model_version not in self._loaded:
            from sentence_transformers import SentenceTransformer

            self._loaded[model_version] = SentenceTransformer(
                self.checkpoints[model_version], trust_remote_code=self.trust_remote_code
            )
        return self._loaded[model_version]

    def embed(self, texts: list[str], model_version: str) -> np.ndarray:
        return self.embed_documents(texts, model_version)

    def embed_documents(self, texts: list[str], model_version: str) -> np.ndarray:
        model = self._get_model(model_version)
        prefixed = [self.document_prefix + t for t in texts]
        return np.asarray(model.encode(prefixed, normalize_embeddings=True), dtype=np.float32)

    def embed_queries(self, texts: list[str], model_version: str) -> np.ndarray:
        model = self._get_model(model_version)
        prefixed = [self.query_prefix + t for t in texts]
        return np.asarray(model.encode(prefixed, normalize_embeddings=True), dtype=np.float32)


def get_backend(name: str, **kwargs) -> EmbeddingBackend:
    if name == "hashing":
        return HashingDriftBackend(**kwargs)
    if name == "sentence-transformer":
        return SentenceTransformerBackend(**kwargs)
    raise ValueError(f"unknown embedding backend: {name}")
