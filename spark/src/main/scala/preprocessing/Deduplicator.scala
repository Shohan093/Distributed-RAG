package preprocessing

import org.apache.spark.sql.DataFrame

object Deduplicator {

  def deduplicate(df: DataFrame): DataFrame = {
    df.dropDuplicates("title", "text")
  }
}