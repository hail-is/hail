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

object Settings {
  val hailMajorMinorVersion = "0.2"
  val hailPatchVersion = "127"
}

/** Update the millw script. */
def millw(): Command[PathRef] = T.command {
  val target =
    mill.util.Util.download("https://raw.githubusercontent.com/lefou/millw/main/millw")
  val millw = T.workspace / "millw"
  os.copy.over(target.path, millw)
  os.perms.set(millw, os.perms(millw) + java.nio.file.attribute.PosixFilePermission.OWNER_EXECUTE)
  target
}

def scalaVersion: T[String] = T.input {
  val v = T.ctx().env.getOrElse("SCALA_VERSION", "2.12.15")
  if (!v.startsWith("2.12"))
    Result.Failure("Hail currently supports only Scala 2.12")
  else
    v
}

def javaVersion: T[String] = T.input {
  System.getProperty("java.version")
}

def sparkVersion: T[String] = T.input {
  Result.Success(T.ctx().env.getOrElse("SPARK_VERSION", "3.3.0"))
}

def debugMode: T[Boolean] = T.input {
  !T.ctx().env.contains("HAIL_RELEASE_MODE")
}

def debugOrRelease: Task[String] = T.task {
  if (debugMode()) "debug" else "release"
}

def buildInfo: T[PathRef] = T {
  val revision = VcsVersion.vcsState().currentRevision
  os.write(
    T.dest / "build-info.properties",
    s"""[Build Metadata]
       |revision=$revision
       |sparkVersion=${sparkVersion()}
       |hailPipVersion=${Settings.hailMajorMinorVersion}.${Settings.hailPatchVersion}
       |""".stripMargin,
  )
  PathRef(T.dest)
}

object Deps {
  object HTTPComponents {
    val core = ivy"org.apache.httpcomponents:httpcore:4.4.14"
    val client = ivy"org.apache.httpcomponents:httpclient:4.5.13"
  }

  object Asm {
    val version: String = "7.3.1"
    val core = ivy"org.ow2.asm:asm:$version"
    val analysis = ivy"org.ow2.asm:asm-analysis:$version"
    val util = ivy"org.ow2.asm:asm-util:$version"
  }

  object Breeze {
    val core = ivy"org.scalanlp::breeze:1.1"
    val natives = ivy"org.scalanlp::breeze-natives:1.1"
  }

  object Commons {
    val io = ivy"commons-io:commons-io:2.11.0"
    val lang3 = ivy"org.apache.commons:commons-lang3:3.12.0"
    val codec = ivy"commons-codec:commons-codec:1.15"
  }

  object Spark {
    def core: Task[Dep] = T.task(ivy"org.apache.spark::spark-core:${sparkVersion()}")
    def mllib: Task[Dep] = T.task(ivy"org.apache.spark::spark-mllib:${sparkVersion()}")
  }

  val samtools = ivy"com.github.samtools:htsjdk:3.0.5"
  val jdistlib = ivy"net.sourceforge.jdistlib:jdistlib:0.4.5"
  val freemarker = ivy"org.freemarker:freemarker:2.3.31"
  val elasticsearch = ivy"org.elasticsearch::elasticsearch-spark-30:8.4.3"
  val gcloud = ivy"com.google.cloud:google-cloud-storage:2.30.1"
  val jna = ivy"net.java.dev.jna:jna:5.13.0"
  val json4s = ivy"org.json4s::json4s-jackson:3.7.0-M11"
  val zstd = ivy"com.github.luben:zstd-jni:1.5.5-11"
  val lz4 = ivy"org.lz4:lz4-java:1.8.0"
  val netlib = ivy"com.github.fommil.netlib:all:1.1.2"
  val avro = ivy"org.apache.avro:avro:1.11.2"
  val junixsocket = ivy"com.kohlschutter.junixsocket:junixsocket-core:2.6.1"
  val log4j = ivy"org.apache.logging.log4j:log4j-1.2-api:2.17.2"
  val hadoopClient = ivy"org.apache.hadoop:hadoop-client:3.3.4"
  val jackson = ivy"com.fasterxml.jackson.core:jackson-core:2.14.2"

  object Plugins {
    val betterModadicFor = ivy"com.olegpy::better-monadic-for:0.3.1"
  }
}

trait HailScalaModule extends SbtModule with ScalafmtModule with ScalafixModule { outer =>
  override def scalaVersion: T[String] = build.scalaVersion()

  override def javacOptions: T[Seq[String]] = Seq(
    "-Xlint:all",
    "-Werror",
    if (debugMode()) "-g" else "-O",
  ) ++ (if (!javaVersion().startsWith("1.8")) Seq("-Xlint:-processing") else Seq())

  override def scalacOptions: T[Seq[String]] = T {
    Seq(
      "-explaintypes",
      "-unchecked",
      "-Xsource:2.13",
      "-Xno-patmat-analysis",
      "-Ypartial-unification",
      "-Yno-adapted-args", // will be removed in 2.13
      "-Xlint",
      "-Ywarn-unused:_,-explicits,-implicits",
    ) ++ (
      if (debugMode()) Seq()
      else Seq(
        "-Xfatal-warnings",
        "-opt:l:method",
        "-opt:-closure-invocations",
      )
    )
  }

  // needed to force IntelliJ to include resources in the classpath when running tests
  override def bspCompileClasspath: T[Agg[UnresolvedPath]] =
    super.bspCompileClasspath() ++ resources().map(p => UnresolvedPath.ResolvedPath(p.path))

  trait HailTests extends SbtModuleTests with TestNg with ScalafmtModule {
    override def forkArgs: T[Seq[String]] = Seq("-Xss4m", "-Xmx4096M")

    override def ivyDeps: T[Agg[Dep]] =
      super.ivyDeps() ++ outer.compileIvyDeps() ++ Agg(
        ivy"org.scalatest::scalatest:3.0.5",
        // testng 7.6 and later does not support java8
        ivy"org.testng:testng:7.5.1",
      )

    // needed to force IntelliJ to include resources in the classpath when running tests
    override def bspCompileClasspath: T[Agg[UnresolvedPath]] =
      super.bspCompileClasspath() ++ resources().map(p => UnresolvedPath.ResolvedPath(p.path))
  }
}

object main extends RootModule with HailScalaModule { outer =>
  override def moduleDeps: Seq[JavaModule] = Seq(memory)

  override def resources: T[Seq[PathRef]] = Seq(
    PathRef(millSourcePath / "src" / "main" / "resources"),
    PathRef(millSourcePath / "prebuilt" / "lib"),
    buildInfo(),
  )

  override def unmanagedClasspath: T[Agg[PathRef]] =
    Agg(shadedazure.assembly())

  // omit unmanagedClasspath from the jar
  override def jar: T[PathRef] =
    Jvm.createJar((resources() ++ Agg(compile().classes)).map(_.path).filter(os.exists), manifest())

  override def ivyDeps: T[Agg[Dep]] = Agg(
    Deps.HTTPComponents.core,
    Deps.HTTPComponents.client,
    Deps.Asm.core,
    Deps.Asm.analysis,
    Deps.Asm.util,
    Deps.samtools.excludeOrg("*"),
    Deps.jdistlib.excludeOrg("*"),
    Deps.freemarker,
    Deps.elasticsearch.excludeOrg("org.apache.spark"),
    Deps.gcloud.excludeOrg("com.fasterxml.jackson.core"),
    Deps.jna,
    Deps.json4s.excludeOrg("com.fasterxml.jackson.core"),
    Deps.zstd,
  )

  override def runIvyDeps: T[Agg[Dep]] = Agg(
    Deps.Breeze.natives.excludeOrg("org.apache.commons.math3"),
    Deps.Commons.io,
    Deps.Commons.lang3,
    //    ivy"org.apache.commons:commons-math3:3.6.1",
    Deps.Commons.codec,
    Deps.lz4,
    Deps.netlib,
    Deps.avro.excludeOrg("com.fasterxml.jackson.core"),
    Deps.junixsocket,
//    Deps.zstd
  )

  override def compileIvyDeps: T[Agg[Dep]] = Agg(
    Deps.log4j,
    Deps.hadoopClient,
    Deps.Spark.core(),
    Deps.Spark.mllib(),
    Deps.Breeze.core,
    //      ivy"org.scalanlp::breeze-natives:1.1",
  )

  override def assemblyRules: Seq[Rule] = super.assemblyRules ++ Seq(
    Rule.Exclude("META-INF/INDEX.LIST"),
    Rule.ExcludePattern("scala/.*"),
    Rule.AppendPattern("META-INF/services/.*", "\n"),
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

  override def scalacPluginIvyDeps: T[Agg[Dep]] = Agg(
    Deps.Plugins.betterModadicFor
  )

  def writeRunClasspath: T[PathRef] = T {
    os.write(
      T.dest / "runClasspath",
      runClasspath().map(_.path).mkString(":"),
    )
    PathRef(T.dest)
  }

  object memory extends JavaModule { // with CrossValue {
    override def zincIncrementalCompilation: T[Boolean] = false

    override def javacOptions: T[Seq[String]] =
      outer.javacOptions() ++ (
        if (javaVersion().startsWith("1.8")) Seq(
          "-XDenableSunApiLintControl",
          "-Xlint:-sunapi",
        )
        else Seq()
      )

    override def sources: T[Seq[PathRef]] = T.sources {
      Seq(PathRef(this.millSourcePath / os.up / "src" / debugOrRelease() / "java"))
    }

    override def compileIvyDeps: T[Agg[Dep]] = Agg(
      Deps.hadoopClient,
      Deps.samtools.excludeOrg("*"),
    )
  }

  object test extends HailTests {
    override def resources: T[Seq[PathRef]] = outer.resources() ++ super.resources()

    override def assemblyRules: Seq[Rule] = outer.assemblyRules ++ Seq(
      Rule.Relocate("org.codehaus.jackson.**", "is.hail.relocated.@0")
//      Rule.Relocate("org.codehaus.stax2.**", "is.hail.relocated.@0"),
    )

    override def ivyDeps: T[Agg[Dep]] = super.ivyDeps() ++ Seq(
      Deps.jackson
    )
  }

  object shadedazure extends JavaModule {
    override def ivyDeps: T[Agg[Dep]] = Agg(
      ivy"com.azure:azure-storage-blob:12.22.0",
      ivy"com.azure:azure-core-http-netty:1.13.7",
      ivy"com.azure:azure-identity:1.8.3",
    )

    override def assemblyRules: Seq[Rule] = Seq(
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

}
