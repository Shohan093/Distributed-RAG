ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
			name := "drift-rag-spark",

			fork := true,

			javaOptions ++= Seq(
				"--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
			),

			libraryDependencies ++= Seq(
				"org.apache.spark" %% "spark-core" % "3.5.6" % "provided",
				"org.apache.spark" %% "spark-sql"  % "3.5.6" % "provided"
			)
		)
