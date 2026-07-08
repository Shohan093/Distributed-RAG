package preprocessing

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object Cleaner {

  def clean(df: DataFrame): DataFrame = {
    df
      // Strip AP wire transmission artifacts like "^By STEVEN WINE=" and "^AP Sports Writer="
      .withColumn("text", regexp_replace(col("text"), "\\^[^\\n]*=", ""))
      // Collapse multiple whitespace/newlines into single spaces for consistency
      .withColumn("text", regexp_replace(col("text"), "\\s+", " "))
      .withColumn("text", trim(col("text")))
      // Drop degenerate rows: very short "articles" that are likely just a stub or headline
      .filter(length(col("text")) >= 200)
  }
}