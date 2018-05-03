import Dependencies._

lazy val spark = "2.2.0"
lazy val si = spark match {
  case "2.2.0" => SparkInfo("0.10.4", "0.13.1")
  case "2.1.0" => SparkInfo("0.10.4", "0.12")
  case "2.0.2" => SparkInfo("0.10.3", "0.11.2")
}

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "is.hail",
      scalaVersion := "2.11.8",
      version      := "0.2.0-SNAPSHOT"
    )),
    name := "hail",
    spName := "hail-is/hail",
    sparkVersion := spark,
    sparkComponents ++= Seq("sql", "mllib"),
    resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
    libraryDependencies ++= Seq(
      scalaTest % Test
        , "org.ow2.asm" % "asm" % "5.1"
        , "org.ow2.asm" % "asm-util" % "5.1"
        , "org.ow2.asm" % "asm-analysis" % "5.1"
        , hadoopClient
        , "net.jpountz.lz4" % "lz4" % "1.3.0"
        , "org.scalanlp" %% "breeze-natives" % si.breezeVersion
        , "com.github.samtools" % "htsjdk" % "2.14.2"
        , "org.slf4j" % "slf4j-api" % "1.7.25"
        , "org.http4s" %% "http4s-core" % "0.12.3"
        , "org.http4s" %% "http4s-server" % "0.12.3"
        , "org.http4s" %% "http4s-argonaut" % "0.12.3"
        , "org.http4s" %% "http4s-dsl" % "0.12.3"
        , "org.http4s" %% "http4s-scala-xml" % "0.12.3"
        , "org.http4s" %% "http4s-client" % "0.12.3"
        , "org.http4s" %% "http4s-websocket" % "0.1.3"
        , "org.http4s" %% "http4s-blaze-core" % "0.12.3"
        , "org.http4s" %% "http4s-blaze-client" % "0.12.3"
        , "org.http4s" %% "http4s-blaze-server" % "0.12.3"
        , "org.json4s" %% "json4s-core" % "3.2.10"
        , "org.json4s" %% "json4s-jackson" % "3.2.10"
        , "org.json4s" %% "json4s-ast" % "3.2.10"
        , "org.elasticsearch" % "elasticsearch-spark-20_2.11" % "6.2.4"
        , "org.apache.solr" % "solr-solrj" % "6.2.0"
        , "com.datastax.cassandra" % "cassandra-driver-core" % "3.0.0"
        , "com.jayway.restassured" % "rest-assured" % "2.8.0"
        , "net.java.dev.jna" % "jna" % "4.2.2"
        , "net.sourceforge.jdistlib" % "jdistlib" % "0.4.5"
        , "org.apache.commons" % "commons-math3" % "3.6.1"
        , "org.testng" % "testng" % "6.8.21" % Test
    ),
    assemblyShadeRules in assembly := Seq(
      ShadeRule
        .rename("org.objectweb.asm.**" -> "shaded.@1")
        .inLibrary(hadoopClient)
    )
  )
