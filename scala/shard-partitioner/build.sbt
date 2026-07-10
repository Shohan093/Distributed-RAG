name := "shard-partitioner"
organization := "com.driftdrag"
version := "0.1.0"
scalaVersion := "2.12.18"

val sparkVersion = "3.5.3"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion,
  "redis.clients" % "jedis" % "5.1.3",
  "org.scalatest" %% "scalatest" % "3.2.18" % Test
)

// Same Spark minor version as the docker-compose spark-master/spark-worker
// images (see ../../docker-compose.yml) and close to the pyspark version
// pinned in ../../requirements.txt -- keep these three in lockstep when
// bumping Spark, or spark-submit will reject the jar on a real cluster.

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case _                             => MergeStrategy.first
}

assembly / mainClass := Some("com.driftdrag.ingest.ShardPartitionerJob")
