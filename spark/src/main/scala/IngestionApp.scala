import org.apache.spark.sql.SparkSession

object IngestionApp extends App {

  val spark = SparkSession.builder()
    .appName("Distributed RAG")
    .master("local[*]")
    .getOrCreate()

  // Reduce log noise so our own output is easier to spot
  spark.sparkContext.setLogLevel("WARN")

  val df = spark.read.parquet("data/raw/cc_news-train-0000.parquet")

  println("=== SCHEMA ===")
  df.printSchema()

  println("=== ROW COUNT ===")
  println(df.count())

  println("=== SAMPLE ROWS (title, date, domain only) ===")
  df.select("title", "date", "domain").show(5, truncate = 60)

  spark.stop()
}
