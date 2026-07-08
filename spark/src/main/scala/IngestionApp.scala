import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import preprocessing.{Cleaner, Deduplicator, Chunker}

object IngestionApp extends App {

  val spark = SparkSession.builder()
    .appName("Distributed RAG")
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")
  import spark.implicits._

  for (epoch <- 1 to 3) {
    println(s"=== PROCESSING EPOCH $epoch ===")

    val raw = spark.read.parquet(s"data/generation$epoch/raw_tagged")
    val rawCount = raw.count()

    val cleaned = Cleaner.clean(raw)
    val cleanedCount = cleaned.count()
    val deduped = Deduplicator.deduplicate(cleaned)
    val truncated = Chunker.truncate(deduped)
    val dedupedCount = truncated.count()

    println(s"Epoch $epoch: raw=$rawCount, after cleaning=$cleanedCount, after dedup+truncate=$dedupedCount")

    truncated.write.mode("overwrite").parquet(s"data/generation$epoch/cleaned")
  }

  println("All epochs processed.")
  spark.stop()
}