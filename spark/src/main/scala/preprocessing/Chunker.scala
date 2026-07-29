package preprocessing

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object Chunker {

  private val maxWords = 400
  private val maxChars = 3000

  private val truncateUdf = udf { (text: String) =>
    if (text == null) {
      text
    } else {
      val words = text.split("\\s+")
      val wordTruncated = if (words.length <= maxWords) text else words.take(maxWords).mkString(" ")
      if (wordTruncated.length <= maxChars) wordTruncated else wordTruncated.take(maxChars)
    }
  }

  def truncate(df: DataFrame): DataFrame = {
    df.withColumn("text", truncateUdf(col("text")))
  }
}
