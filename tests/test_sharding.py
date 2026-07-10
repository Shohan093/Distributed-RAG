from driftdrag.ingest.sharding import shard_for_doc


def test_matches_scala_crc32_fixtures():
    # Cross-checked against java.util.zip.CRC32 in
    # scala/shard-partitioner/src/test/scala/.../ShardPartitioningSpec.scala
    assert shard_for_doc("doc-12345", 32) == 0
    assert shard_for_doc("doc-1", 32) == 9
    assert shard_for_doc("doc-999999", 32) == 25


def test_deterministic_and_in_range():
    num_shards = 16
    for i in range(1000):
        doc_id = f"doc-{i}"
        shard = shard_for_doc(doc_id, num_shards)
        assert 0 <= shard < num_shards
        assert shard == shard_for_doc(doc_id, num_shards)
