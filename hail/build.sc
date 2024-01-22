import $ivy.`com.goyeau::mill-scalafix::0.3.2`
import $ivy.`de.tototec::de.tobiasroeser.mill.vcs.version::0.4.0`
import com.goyeau.mill.scalafix.ScalafixModule
import de.tobiasroeser.mill.vcs.version.VcsVersion
import mill._
import mill.api.Result
import mill.scalalib._
import mill.scalalib.Assembly._
import mill.scalalib.TestModule.TestNg
import mill.scalalib.scalafmt.ScalafmtModule
import mill.util.Jvm

def hailMajorMinorVersion = "0.2"
def hailPatchVersion = "127"

/** Update the millw script. */
def millw() = T.command {
  val target =
    mill.util.Util.download("https://raw.githubusercontent.com/lefou/millw/main/millw")
  val millw = T.workspace / "millw"
  os.copy.over(target.path, millw)
  os.perms.set(millw, os.perms(millw) + java.nio.file.attribute.PosixFilePermission.OWNER_EXECUTE)
  target
}

def scalaVersion = T.input {
  val v = T.ctx().env.getOrElse("SCALA_VERSION", "2.12.15")
  if (!v.startsWith("2.12"))
    Result.Failure("Hail currently supports only Scala 2.12")
  else
    v
}

def javaVersion = T.input {
  System.getProperty("java.version")
}

def sparkVersion = T.input {
  T.ctx().env.getOrElse("SPARK_VERSION", "3.3.0")
}

def buildInfo: T[PathRef] = T {
  val revision = VcsVersion.vcsState().currentRevision
  os.write(
    T.dest / "build-info.properties",
    s"""[Build Metadata]
       |revision=$revision
       |sparkVersion=${sparkVersion()}
       |hailPipVersion=$hailMajorMinorVersion.$hailPatchVersion
       |""".stripMargin,
  )
  PathRef(T.dest)
}

object shadedazure extends JavaModule {
  override def ivyDeps = Agg(
    ivy"com.azure:azure-storage-blob:12.22.0",
    ivy"com.azure:azure-core-http-netty:1.13.7",
    ivy"com.azure:azure-identity:1.8.3",
  )

  override def assemblyRules = Seq(
    Rule.ExcludePattern("META-INF/*.RSA"),
    Rule.ExcludePattern("META-INF/*.SF"),
    Rule.ExcludePattern("META-INF/*.DSA"),
    Rule.Relocate("com.azure.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("com.ctc.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("com.fasterxml.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("com.microsoft.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("com.nimbusds.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("com.sun.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("io.netty.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("is.hail.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("net.jcip.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("net.minidev.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("org.apache.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("org.codehaus.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("org.objectweb.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("org.reactivestreams.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("org.slf4j.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("reactor.adapter.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("reactor.core.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("reactor.netty.**", "is.hail.shadedazure.@0"),
    Rule.Relocate("reactor.util.**", "is.hail.shadedazure.@0"),
  )
}

trait HailScalaModule extends SbtModule with ScalafmtModule with ScalafixModule { outer =>
  def crossValue: String
  def debugBuild: Boolean = crossValue == "debug"

  override def scalaVersion: T[String] = build.scalaVersion()

  def asmVersion = "7.3.1"

  override def javacOptions: T[Seq[String]] = Seq(
    "-Xlint:all",
    "-Werror",
    if (debugBuild) "-g" else "-O",
  ) ++ (if (!javaVersion().startsWith("1.8")) Seq("-Xlint:-processing") else Seq())

  override def scalacOptions: T[Seq[String]] = T {
    Seq(
      "-Xsource:2.13",
      "-Xno-patmat-analysis",
      "-Ypartial-unification",
      "-Yno-adapted-args", // will be removed in 2.13
    ) ++ (
      if (debugBuild) Seq(
        "-explaintypes",
        "-unchecked",
        "-Xlint",
        "-Ywarn-unused:_,-explicits,-implicits",
        "-Wconf:cat=unused-locals:w,cat=unused:info,any:ws",
      ) else Seq(
        "-opt:l:method",
        "-opt:-closure-invocations",
      )
    )
  }

  def printRunClasspath(): Command[Unit] = T.command {
    println(runClasspath().map(_.path).mkString(":"))
  }

  trait HailTests extends SbtModuleTests with TestNg with ScalafmtModule {
    // FIXME: remove when warnings are fixed in tests
    override def scalacOptions = outer.scalacOptions() ++ Seq(
      "-Ywarn-unused:-patvars,-locals,-privates",
      "-Xlint:-infer-any,-nullary-unit,-nullary-override",
    )

    override def forkArgs = Seq("-Xss4m", "-Xmx4096M")

    override def ivyDeps =
      super.ivyDeps() ++ outer.compileIvyDeps() ++ Agg(
        ivy"org.scalatest::scalatest:3.0.5",
        // testng 7.6 and later does not support java8
        ivy"org.testng:testng:7.5.1",
      )
  }
}

object main extends Cross[MainModule]("debug", "release")

trait MainModule extends Cross.Module[String] with HailScalaModule { outer =>
  override def millSourcePath = millOuterCtx.millSourcePath / os.up

  override def moduleDeps = Seq(memory)

  override def resources = T {
    super.resources() ++ Seq(
      buildInfo(),
      PathRef(millSourcePath / "prebuilt" / "lib"),
    )
  }

  override def unmanagedClasspath = T {
    Agg(shadedazure.assembly())
  }

  // omit unmanagedClasspath from the jar
  override def jar: T[PathRef] = T {
    Jvm.createJar((resources() ++ Agg(compile().classes)).map(_.path).filter(os.exists), manifest())
  }

  override def ivyDeps = Agg(
    ivy"org.apache.httpcomponents:httpcore:4.4.14",
    ivy"org.apache.httpcomponents:httpclient:4.5.13",
    ivy"org.ow2.asm:asm:$asmVersion",
    ivy"org.ow2.asm:asm-analysis:$asmVersion",
    ivy"org.ow2.asm:asm-util:$asmVersion",
    ivy"com.github.samtools:htsjdk:3.0.5".excludeOrg("*"),
    ivy"net.sourceforge.jdistlib:jdistlib:0.4.5".excludeOrg("*"),
    ivy"org.freemarker:freemarker:2.3.31",
    ivy"org.elasticsearch::elasticsearch-spark-30:8.4.3".excludeOrg("org.apache.spark"),
    ivy"com.google.cloud:google-cloud-storage:2.30.1".excludeOrg("com.fasterxml.jackson.core"),
    ivy"net.java.dev.jna:jna:5.13.0",
    ivy"org.json4s::json4s-jackson:3.7.0-M11".excludeOrg("com.fasterxml.jackson.core"),
    ivy"com.github.luben:zstd-jni:1.5.5-11",
  )

  override def runIvyDeps = Agg(
    ivy"org.scalanlp::breeze-natives:1.1".excludeOrg("org.apache.commons.math3"),
    ivy"commons-io:commons-io:2.11.0",
    ivy"org.apache.commons:commons-lang3:3.12.0",
    //    ivy"org.apache.commons:commons-math3:3.6.1",
    ivy"commons-codec:commons-codec:1.15",
    ivy"org.lz4:lz4-java:1.8.0",
    ivy"com.github.fommil.netlib:all:1.1.2",
    ivy"org.apache.avro:avro:1.11.2".excludeOrg("com.fasterxml.jackson.core"),
    ivy"com.kohlschutter.junixsocket:junixsocket-core:2.6.1",
    ivy"com.github.luben:zstd-jni:1.5.5-11",
  )

  override def compileIvyDeps = Agg(
    ivy"org.apache.logging.log4j:log4j-1.2-api:2.17.2",
    ivy"org.apache.hadoop:hadoop-client:3.3.4",
    ivy"org.apache.spark::spark-core:${sparkVersion()}",
    ivy"org.apache.spark::spark-mllib:${sparkVersion()}",
    ivy"org.scalanlp::breeze:1.1",
    //      ivy"org.scalanlp::breeze-natives:1.1",
  )

  override def assemblyRules = super.assemblyRules ++ Seq(
    Rule.Exclude("META-INF/INDEX.LIST"),
    Rule.ExcludePattern("scala/.*"),
    Rule.AppendPattern("META-INF/services/.*"),
    Rule.Relocate("breeze.**", "is.hail.relocated.@0"),
    Rule.Relocate("com.google.cloud.**", "is.hail.relocated.@0"),
    Rule.Relocate("com.google.common.**", "is.hail.relocated.@0"),
    Rule.Relocate("org.apache.commons.io.**", "is.hail.relocated.@0"),
    Rule.Relocate("org.apache.commons.lang3.**", "is.hail.relocated.@0"),
    Rule.Relocate("org.apache.http.**", "is.hail.relocated.@0"),
    Rule.Relocate("org.elasticsearch.**", "is.hail.relocated.@0"),
    Rule.Relocate("org.json4s.**", "is.hail.relocated.@0"),
    Rule.Relocate("org.objectweb.**", "is.hail.relocated.@0"),
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"com.olegpy::better-monadic-for:0.3.1"
  )

  object memory extends JavaModule with CrossValue {
    override def zincIncrementalCompilation = false

    override def javacOptions: T[Seq[String]] =
      outer.javacOptions() ++ (
        if (javaVersion().startsWith("1.8")) Seq(
          "-XDenableSunApiLintControl",
          "-Xlint:-sunapi",
        ) else Seq()
      )

    override def sources = T.sources {
      Seq(PathRef(this.millSourcePath / os.up / "src" / crossValue / "java"))
    }

    override def compileIvyDeps = Agg(
      ivy"org.apache.hadoop:hadoop-client:3.3.4",
      ivy"com.github.samtools:htsjdk:3.0.5".excludeOrg("*"),
    )
  }

  object test extends HailTests {
    override def resources = outer.resources() ++ super.resources()

    override def assemblyRules = outer.assemblyRules ++ Seq(
      Rule.Relocate("org.codehaus.jackson.**", "is.hail.relocated.@0")
//      Rule.Relocate("org.codehaus.stax2.**", "is.hail.relocated.@0"),
    )

    override def ivyDeps = super.ivyDeps() ++ Seq(
      ivy"com.fasterxml.jackson.core:jackson-core:2.14.2"
    )
  }
}
