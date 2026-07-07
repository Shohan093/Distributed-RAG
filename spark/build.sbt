ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "2.13.18"

lazy val root = (project in file("."))
  .settings(
    name := "drift-rag-spark"
  )
