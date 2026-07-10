"""Canonical shard-assignment function.

Both the PySpark embedding job (Python, this module) and the Scala shard
partitioner (scala/shard-partitioner) must map a given doc_id to the same
shard_id independently -- they run as separate streaming jobs consuming the
same Kafka topic, with no RPC between them. CRC32 is the join point: Python's
``zlib.crc32`` and Java/Scala's ``java.util.zip.CRC32`` both implement the
identical CRC-32 (IEEE 802.3) algorithm over the same input bytes, so hashing
``doc_id.encode("utf-8")`` in either language yields the same 32-bit value
and therefore the same ``value % num_shards``. Do not swap this for
Python's builtin ``hash()`` (salted per-process) or Scala's ``.hashCode``
(JVM-specific, not equal to Python's) -- either breaks cross-language
agreement silently.
"""

from __future__ import annotations

import zlib


def shard_for_doc(doc_id: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    checksum = zlib.crc32(doc_id.encode("utf-8"))
    return checksum % num_shards
