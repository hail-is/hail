import java.nio.file.Paths
import scala.sys.process._

organization  := "is.hail"

version       := "0.1"

scalaVersion  := "2.11.8"

scalacOptions := Seq(
  "-Xfatal-warnings",
  "-Xlint:_",
  "-deprecation",
  "-unchecked",
  "-Xlint:-infer-any",
  "-Xlint:-unsound-match",
  "-encoding",
  "utf8"
)

lazy val compileHail = taskKey[Unit]("compile hail")
compileHail := {
  if ((Seq("/bin/sh", "-c", "cd ../hail && ./gradlew jar") ! streams.value.log) != 0) {
    throw new sbt.FeedbackProvidedException() {
      override def toString() = "`cd ../hail && ./gradlew jar` returned non-zero exit-code" }
  }
}

(compile in Compile) := ((compile in Compile) dependsOn compileHail).value

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

libraryDependencies ++= {
  val sparkVersion = "2.4.2"
  Seq(
      "com.typesafe.akka" %% "akka-http"   % "10.1.9"
    , "com.typesafe.akka" %% "akka-stream" % "2.5.23"
    , "is.hail" % "Hail" % "0.1" from s"file://${Paths.get("..").toAbsolutePath}/hail/build/libs/hail.jar"
    , "log4j" % "log4j" % "1.2.17"
    , "org.scalanlp" %% "breeze" % "0.13.2"
    , "org.apache.spark" %% "spark-mllib" % sparkVersion
    , "org.ow2.asm" % "asm" % "5.1"
    , "org.ow2.asm" % "asm-util" % "5.1"
    , "org.ow2.asm" % "asm-analysis" % "5.1"

    // hail deps
    // , "org.apache.spark" %% "spark-mllib" % sparkVersion
    // , "org.apache.spark" %% "spark-sql" % sparkVersion
    // , "org.ow2.asm" % "asm" % "5.1"
    // , "org.ow2.asm" % "asm-util" % "5.1"
    // , "org.ow2.asm" % "asm-analysis" % "5.1"
    // , "org.apache.hadoop" % "hadoop-client" % "2.7.1"
    // , "net.jpountz.lz4" % "lz4" % "1.3.0"
    // , "org.scalanlp" %% "breeze-natives" % "0.13.2"
    // , "com.github.samtools" % "htsjdk" % "2.18.0"
    // , "org.slf4j" % "slf4j-api" % "1.7.25"
    // , "org.http4s" %% "http4s-core" % "0.12.3"
    // , "org.http4s" %% "http4s-server" % "0.12.3"
    // , "org.http4s" %% "http4s-argonaut" % "0.12.3"
    // , "org.http4s" %% "http4s-dsl" % "0.12.3"
    // , "org.http4s" %% "http4s-scala-xml" % "0.12.3"
    // , "org.http4s" %% "http4s-client" % "0.12.3"
    // , "org.http4s" %% "http4s-websocket" % "0.1.3"
    // , "org.http4s" %% "http4s-blaze-core" % "0.12.3"
    // , "org.http4s" %% "http4s-blaze-client" % "0.12.3"
    // , "org.http4s" %% "http4s-blaze-server" % "0.12.3"
    // , "org.json4s" %% "json4s-core" % "3.2.10"
    // , "org.json4s" %% "json4s-jackson" % "3.2.10"
    // , "org.json4s" %% "json4s-ast" % "3.2.10"
    // , "org.elasticsearch" % "elasticsearch-spark-20_2.11" % "6.2.4"
    // , "org.apache.solr" % "solr-solrj" % "6.2.0"
    // , "com.datastax.cassandra" % "cassandra-driver-core" % "3.0.0"
    // , "com.jayway.restassured" % "rest-assured" % "2.8.0"
    // , "net.java.dev.jna" % "jna" % "4.2.2"
    // , "net.sourceforge.jdistlib" % "jdistlib" % "0.4.5"
    // , "org.apache.commons" % "commons-math3" % "3.6.1"
  )
}

mainClass in assembly := Some("is.hail.shuffler.WebServer")

assemblyMergeStrategy in assembly := {
 case PathList("META-INF" | "git.properties", xs @ _*) => MergeStrategy.discard
 case PathList("reference.conf", xs @ _*) => MergeStrategy.concat
 case x => MergeStrategy.first
}
