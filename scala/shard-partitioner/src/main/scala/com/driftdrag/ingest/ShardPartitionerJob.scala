package com.driftdrag.ingest

import java.nio.charset.StandardCharsets
import java.util.zip.CRC32

import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import redis.clients.jedis.JedisPool

/** Canonical shard assignment, mirrored in Python at
  * src/driftdrag/ingest/sharding.py. Both languages hash with CRC-32
  * (IEEE 802.3 / the same polynomial zlib and java.util.zip.CRC32 both
  * implement) over the UTF-8 bytes of doc_id, so a document routes to the
  * same shard whether it was assigned by this Scala job or the PySpark
  * embedding job -- the two run as independent streaming consumers of the
  * same Kafka topic with no RPC between them, so agreement has to come from
  * the hash function alone. Do NOT swap this for Scala's `.hashCode` or
  * Python's builtin `hash()`; neither matches across languages/processes.
  */
object ShardPartitioning {
  def shardFor(docId: String, numShards: Int): Int = {
    val crc = new CRC32()
    crc.update(docId.getBytes(StandardCharsets.UTF_8))
    (crc.getValue % numShards).toInt
  }
}

/** Stage 0 sharding layer, the Scala/Spark half of the polyglot ingest path
  * (see PLAN.md "Big-Data Framework Mapping"): high-throughput Kafka
  * consumption, shard assignment, and Drift Ledger bookkeeping are done
  * here in Scala for direct, JVM-native access to the Kafka connector and
  * to Jedis; the PySpark job (ingest/spark_embed_job.py) owns the parts of
  * Stage 0 that need the Python ML ecosystem (embedding models). Both
  * consume the same topic independently and write to the same data lake
  * root, partitioned by shard_id/model_version, so either job can be
  * scaled or restarted without the other.
  */
case class JobConfig(
    source: String = "kafka",
    bootstrapServers: String = "localhost:9092",
    topic: String = "docs-raw",
    inputPath: String = "data/stream_in",
    outputPath: String = "data/lake/shard_assignments",
    checkpointPath: String = "data/lake/_checkpoints/shard_partitioner",
    numShards: Int = 32,
    modelVersion: String = "v0",
    redisHost: String = "localhost",
    redisPort: Int = 6379,
    triggerOnce: Boolean = true
)

object ShardPartitionerJob {

  val docSchema: StructType = StructType(
    Seq(
      StructField("doc_id", StringType, nullable = false),
      StructField("text", StringType, nullable = false),
      StructField("timestamp", LongType, nullable = true),
      StructField("source", StringType, nullable = true)
    )
  )

  def parseArgs(args: Array[String]): JobConfig =
    args.foldLeft(JobConfig()) { (cfg, arg) =>
      arg.split("=", 2) match {
        case Array("--source", v)            => cfg.copy(source = v)
        case Array("--bootstrap-servers", v) => cfg.copy(bootstrapServers = v)
        case Array("--topic", v)             => cfg.copy(topic = v)
        case Array("--input-path", v)        => cfg.copy(inputPath = v)
        case Array("--output-path", v)       => cfg.copy(outputPath = v)
        case Array("--checkpoint-path", v)   => cfg.copy(checkpointPath = v)
        case Array("--num-shards", v)        => cfg.copy(numShards = v.toInt)
        case Array("--model-version", v)     => cfg.copy(modelVersion = v)
        case Array("--redis-host", v)        => cfg.copy(redisHost = v)
        case Array("--redis-port", v)        => cfg.copy(redisPort = v.toInt)
        case Array("--trigger-once", v)      => cfg.copy(triggerOnce = v.toBoolean)
        case _                               => cfg
      }
    }

  def readDocStream(spark: SparkSession, cfg: JobConfig): DataFrame =
    cfg.source match {
      case "kafka" =>
        val raw = spark.readStream
          .format("kafka")
          .option("kafka.bootstrap.servers", cfg.bootstrapServers)
          .option("subscribe", cfg.topic)
          .option("startingOffsets", "latest")
          .load()
        raw.select(from_json(col("value").cast("string"), docSchema).alias("d")).select("d.*")
      case "file" =>
        spark.readStream
          .format("json")
          .schema(docSchema)
          .option("maxFilesPerTrigger", 1)
          .load(cfg.inputPath)
      case other =>
        throw new IllegalArgumentException(s"unknown source: $other (expected 'kafka' or 'file')")
    }

  def updateLedger(batchDf: DataFrame, cfg: JobConfig): Unit = {
    val perShardCounts = batchDf.groupBy("shard_id").count().collect()
    val pool = new JedisPool(cfg.redisHost, cfg.redisPort)
    try {
      val jedis = pool.getResource
      try
        perShardCounts.foreach { row =>
          val shardId = row.getAs[Int]("shard_id")
          val count = row.getAs[Long]("count")
          jedis.incrBy(s"ledger:shard-$shardId:docs_seen", count)
          jedis.set(s"ledger:shard-$shardId:last_seen_version", cfg.modelVersion)
        }
      finally jedis.close()
    } finally pool.close()
  }

  def main(args: Array[String]): Unit = {
    val cfg = parseArgs(args)

    val spark = SparkSession.builder().appName("drift-drag-shard-partitioner").getOrCreate()

    val shardUdf = udf((docId: String) => ShardPartitioning.shardFor(docId, cfg.numShards))

    val docs = readDocStream(spark, cfg)
      .withColumn("shard_id", shardUdf(col("doc_id")))
      .withColumn("model_version", lit(cfg.modelVersion))

    val writer = docs.writeStream
      .option("checkpointLocation", cfg.checkpointPath)
      .foreachBatch { (batchDf: DataFrame, _: Long) =>
        if (batchDf.take(1).nonEmpty) {
          batchDf.write.mode("append").partitionBy("shard_id", "model_version").parquet(cfg.outputPath)
          updateLedger(batchDf, cfg)
        }
      }

    val query = if (cfg.triggerOnce) writer.trigger(Trigger.AvailableNow()).start() else writer.start()
    query.awaitTermination()
  }
}
