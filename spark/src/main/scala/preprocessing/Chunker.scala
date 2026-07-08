package preprocessing

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object Chunker {

  private val maxWords = 400

  private val truncateUdf = udf { (text: String) =>
    if (text == null) {
      text
    } else {
      val words = text.split("\\s+")
      if (words.length <= maxWords) {
        text
      } else {
        words.take(maxWords).mkString(" ")
      }
    }
  }

  def truncate(df: DataFrame): DataFrame = {
    df.withColumn("text", truncateUdf(col("text")))
  }
}