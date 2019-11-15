import java.nio.file.Paths
import scala.sys.process._

organization  := "is.hail"

version       := "0.1"

scalaVersion  := "2.11.8"

scalacOptions := Seq(
  "-Xfatal-warnings",
  "-Xlint:_",
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Xlint:-infer-any",
  "-Xlint:-unsound-match",
  "-encoding",
  "utf8"
)

lazy val compileHail = taskKey[Unit]("compile hail")
compileHail := {
  if ((Seq("/bin/sh", "-c", "cd ../hail && ./gradlew jar --build-cache") ! streams.value.log) != 0) {
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
  )
}

mainClass in assembly := Some("is.hail.shuffler.WebServer")

assemblyMergeStrategy in assembly := {
 case PathList("META-INF" | "git.properties", xs @ _*) => MergeStrategy.discard
 case PathList("reference.conf", xs @ _*) => MergeStrategy.concat
 case x => MergeStrategy.first
}
