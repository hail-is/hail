package org.broadinstitute.hail.driver

import java.io.File

import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.Main._
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineException, CmdLineParser}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag
import org.apache.log4j.Logger
import org.apache.log4j.Level

object HailConfiguration {
  var installDir: String = _

  var tmpDir: String = _

}

object Main {

  class Options {
    @Args4jOption(required = false, name = "-h", aliases = Array("--help"), usage = "Print usage")
    var printUsage: Boolean = false

    @Args4jOption(required = false, name = "--master", usage = "Set Spark master (default: system default or local[*])")
    var master: String = _

    @Args4jOption(required = false, name = "--noisy", usage = "Enable Spark INFO messages")
    var noisy = false

    @Args4jOption(required = false, name = "--parquet-compression", usage = "Parquet compression codec")
    var parquetCompression = "uncompressed"

    @Args4jOption(required = false, name = "-t", aliases = Array("--tmpdir"), usage = "Temporary directory (default: /tmp)")
    var tmpDir: String = "/tmp"
  }

  def main(args: Array[String]) {

    println("user.dir = " + System.getProperty("user.dir"))

    def splitBefore[T](a: Array[T], p: (T) => Boolean)
      (implicit tct: ClassTag[T]): Array[Array[T]] = {
      val r = mutable.ArrayBuilder.make[Array[T]]()
      val b = mutable.ArrayBuilder.make[T]()
      a.foreach { (x: T) =>
        if (p(x)) {
          r += b.result()
          b.clear()
        }
        b += x
      }
      r += b.result()
      b.clear()
      r.result()
    }
    /*
    def splitBefore[T](a: Array[T], p: (T) => Boolean)
      (implicit tct: ClassTag[T]): Array[Array[T]] =
      a.foldLeft(Array[Array[T]](Array.empty[T])) {
        (acc: Array[Array[T]], s: T) => if (p(s))
          acc :+ Array(s)
        else
          acc.init :+ (acc.last :+ s)
      }
    */
    val commands = Array(
      Cache,
      Count,
      DownsampleVariants,
      ExportGenotypes,
      ExportSamples,
      ExportVariants,
      ExportVCF,
      FilterGenotypes,
      FamSummary,
      FilterVariants,
      FilterSamples,
      GQByDP,
      Import,
      LinearRegressionCommand,
      MendelErrorsCommand,
      PCA,
      Read,
      Repartition,
      SampleQC,
      VariantQC,
      Write
    )

    val nameCommand = commands
      .map(c => (c.name, c))
      .toMap

    val splitArgs: Array[Array[String]] = splitBefore[String](args, (arg: String) => nameCommand.contains(arg))

    val globalArgs = splitArgs(0)

    val options = new Options
    val parser = new CmdLineParser(options)
    try {
      parser.parseArgument((globalArgs: Iterable[String]).asJavaCollection)
      if (options.printUsage) {
        println("usage: hail [<global options>] <cmd1> [<cmd1 args>]")
        println("            [<cmd2> [<cmd2 args>] ... <cmdN> [<cmdN args>]]")
        println("")
        println("global options:")
        new CmdLineParser(new Options).printUsage(System.out)
        println("")
        println("commands:")
        val maxLen = commands.map(_.name.size).max
        commands.foreach(cmd => println("  " + cmd.name + (" " * (maxLen - cmd.name.size + 2))
          + cmd.description))
        sys.exit(0)
      }
    } catch {
      case e: CmdLineException =>
        println(e.getMessage)
        sys.exit(1)
    }

    if (splitArgs.size == 1)
      fatal("no commands given")
    val invocations = splitArgs.tail

    if (!options.noisy) {
      Logger.getLogger("org").setLevel(Level.OFF)
      Logger.getLogger("akka").setLevel(Level.OFF)
    }

    val conf = new SparkConf().setAppName("Hail")
    if (options.master != null)
      conf.setMaster(options.master)
    else if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    conf.set("spark.sql.parquet.compression.codec", options.parquetCompression)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val sc = new SparkContext(conf)

    val hadoopConf = sc.hadoopConfiguration

    // FIXME: when writing to S3, edit configuration files
    hadoopConf.set(
      "spark.sql.parquet.output.committer.class",
      "org.apache.spark.sql.parquet.DirectParquetOutputCommitter")

    hadoopConf.set("io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec,org.broadinstitute.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec")

    val accessKeyID = System.getenv("AWS_ACCESS_KEY_ID")
    if (accessKeyID != null) {
      hadoopConf.set("fs.s3a.access.key", accessKeyID)
      hadoopConf.set("fs.s3n.access.key", accessKeyID)
    }
    val secretAccessKey = System.getenv("AWS_ACCESS_KEY_ID")
    if (secretAccessKey != null) {
      hadoopConf.set("fs.s3a.secret.key", secretAccessKey)
      hadoopConf.set("fs.s3n.secret.key", secretAccessKey)
    }

    // FIXME separate entrypoints
    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    sc.addJar(jar)

    HailConfiguration.installDir = new File(jar).getParent + "/.."
    HailConfiguration.tmpDir = options.tmpDir

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val times = mutable.ArrayBuffer.empty[(String, Long)]

    invocations.foldLeft(State(sc, sqlContext)) { case (s, args) =>
      println("running: " + args.mkString(" "))
      val cmdName = args(0)
      nameCommand.get(cmdName) match {
        case Some(cmd) =>
          val (newS, duration) = time {
            cmd.run(s, args.tail)
          }
          times += cmdName -> duration
          newS
        case None =>
          fatal("unknown command `" + cmdName + "'")
      }
    }

    // Thread.sleep(60*60*1000)

    sc.stop()

    println("timing:")
    times.foreach { case (name, duration) =>
      println("  " + name + ": " + formatTime(duration))
    }
  }
}
