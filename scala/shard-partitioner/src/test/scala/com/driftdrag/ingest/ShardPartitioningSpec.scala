package com.driftdrag.ingest

import org.scalatest.funsuite.AnyFunSuite

/** Fixture values were generated once with Python's `zlib.crc32` (see
  * tests/test_sharding.py) and pinned here so `sbt test` catches any future
  * accidental divergence between the two languages' shard assignment --
  * that agreement is a correctness requirement, not an implementation
  * detail (see ShardPartitioning's doc comment).
  */
class ShardPartitioningSpec extends AnyFunSuite {

  test("shardFor matches the CRC32 value computed independently in Python (zlib.crc32)") {
    assert(ShardPartitioning.shardFor("doc-12345", 32) == 0)
    assert(ShardPartitioning.shardFor("doc-1", 32) == 9)
    assert(ShardPartitioning.shardFor("doc-999999", 32) == 25)
  }

  test("shardFor is deterministic and within range") {
    val numShards = 16
    for (i <- 0 until 1000) {
      val shard = ShardPartitioning.shardFor(s"doc-$i", numShards)
      assert(shard >= 0 && shard < numShards)
      assert(shard == ShardPartitioning.shardFor(s"doc-$i", numShards))
    }
  }
}
