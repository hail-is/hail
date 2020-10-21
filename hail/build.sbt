lazy val sparkVersion = "2.4.0"
lazy val si = sparkVersion match {
  case "3.0.0" => SparkInfo("0.10.9", "1.1")
  case "2.4.2" => SparkInfo("0.10.7", "0.13.2")
  case "2.4.1" => SparkInfo("0.10.7", "0.13.2")
  case "2.4.0" => SparkInfo("0.10.7", "0.13.2")
  case "2.2.0" => SparkInfo("0.10.4", "0.13.1")
  case "2.1.0" => SparkInfo("0.10.4", "0.12")
  case "2.0.2" => SparkInfo("0.10.3", "0.11.2")
}

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "is.hail",
      scalaVersion := "2.12.8",
      version      := "0.2.0-SNAPSHOT"
    )),
    name := "hail",
    resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
    Compile / javacOptions ++= Seq(
      "-Xlint:all"
    ),
    Compile / scalacOptions ++= Seq(
      "-Xfatal-warnings",
      "-Xno-patmat-analysis",
      "-Xlint:_",
      "-deprecation",
      "-unchecked",
      "-Xlint:-infer-any",
      "-Xlint:-unsound-match",
      "-target:jvm-1.8"
    ),
    libraryDependencies ++= Seq(
          "org.scalatest" %% "scalatest" % "3.0.3" % Test
        , "com.google.cloud" % "google-cloud-storage" % "1.106.0"
        , "org.ow2.asm" % "asm" % "5.1"
        , "org.ow2.asm" % "asm-util" % "5.1"
        , "org.ow2.asm" % "asm-analysis" % "5.1"
        , "org.apache.spark" %% "spark-core" % sparkVersion % "provided"
        , "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
        , "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
        , "org.lz4" % "lz4-java" % "1.4.0"
        , "org.scalanlp" %% "breeze-natives" % si.breezeVersion
        , "com.github.samtools" % "htsjdk" % "2.21.0"
        , "org.slf4j" % "slf4j-api" % "1.7.25"
        , "org.elasticsearch" % "elasticsearch-spark-20_2.11" % "6.2.4"
        , "net.java.dev.jna" % "jna" % "4.2.2"
        , "net.sourceforge.jdistlib" % "jdistlib" % "0.4.5"
        , "org.apache.commons" % "commons-math3" % "3.6.1"
        , "org.testng" % "testng" % "6.8.21" % Test
        , "com.indeed" % "lsmtree-core" % "1.0.7"
        , "com.indeed" % "util-serialization" % "1.0.30"
    ),
    unmanagedClasspath in Test += baseDirectory.value / "prebuilt" / "lib"
  )
