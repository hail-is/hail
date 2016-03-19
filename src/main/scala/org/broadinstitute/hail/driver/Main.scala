package org.broadinstitute.hail.driver

import java.io.File
import java.util.Properties
import org.apache.log4j.{LogManager, PropertyConfigurator}
import org.apache.spark.sql.SQLContext
import org.apache.spark._
import org.broadinstitute.hail.FatalException
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods.VCFReport
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineException, CmdLineParser}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

object HailConfiguration {
  var stacktrace: Boolean = _

  var installDir: String = _

  var tmpDir: String = _
}

object Main {

  class Options {
    @Args4jOption(required = false, name = "-a", aliases = Array("--log-append"), usage = "Append to log file")
    var logAppend: Boolean = false

    @Args4jOption(required = false, name = "-h", aliases = Array("--help"), usage = "Print usage")
    var printUsage: Boolean = false

    @Args4jOption(required = false, name = "-l", aliases = Array("--log-file"), usage = "Log file")
    var logFile: String = "hail.log"

    @Args4jOption(required = false, name = "--master", usage = "Set Spark master (default: system default or local[*])")
    var master: String = _

    @Args4jOption(required = false, name = "--parquet-compression", usage = "Parquet compression codec")
    var parquetCompression = "uncompressed"

    @Args4jOption(required = false, name = "-q", aliases = Array("--quiet"), usage = "Don't write log file")
    var logQuiet: Boolean = false

    @Args4jOption(required = false, name = "--stacktrace", usage = "Print stacktrace on exception")
    var stacktrace: Boolean = false

    @Args4jOption(required = false, name = "-t", aliases = Array("--tmpdir"), usage = "Temporary directory")
    var tmpDir: String = "/tmp"
  }

  def logAndPropagateException(cmd: Command, e: Exception): Nothing = {
    log.error(s"${cmd.name}: exception", e)
    if (HailConfiguration.stacktrace)
      throw e
    else {
      System.err.println(s"hail: ${cmd.name}: caught exception: ${e.getClass.getName}: ${e.getMessage}")
      sys.exit(1)
    }
  }

  def handlePropagatedException(cmd: Command,
    exceptionName: String,
    e: Exception) {
    val exceptionStr = exceptionName + ": "
    var pos = e.getMessage.indexOf(exceptionStr)
    if (pos >= 0) {
      val atpos = e.getMessage.indexOf("\n\tat")
      val submsg = e.getMessage.substring(pos + exceptionStr.length, atpos)
      System.err.println(s"hail: ${cmd.name}: fatal: $submsg")
      sys.exit(1)
    }
  }

  def runCommand(s: State, cmd: Command, cmdOpts: Command#Options): State = {
    try {
      cmd.runCommand(s, cmdOpts.asInstanceOf[cmd.Options])
    } catch {
      case f: FatalException =>
        System.err.println(s"hail: ${cmd.name}: fatal: ${f.getMessage}")
        log.error(f.getMessage)
        sys.exit(1)

      case e: SparkException =>
        handlePropagatedException(cmd, "org.broadinstitute.hail.FatalException", e)
        handlePropagatedException(cmd, "org.broadinstitute.hail.PropagatedTribbleException", e)

        // else
        logAndPropagateException(cmd, e)

      case e: Exception =>
        logAndPropagateException(cmd, e)
    }
  }

  def runCommands(sc: SparkContext,
    sqlContext: SQLContext,
    invocations: Array[(Command, Command#Options, Array[String])]) {

    val times = mutable.ArrayBuffer.empty[(String, Long)]

    invocations.foldLeft(State(sc, sqlContext)) { case (s, (cmd, cmdOpts, cmdArgs)) =>
      info(s"running: ${
        cmdArgs
          .map { s => if (s.contains(" ")) s"'$s'" else s }
          .mkString(" ")
      }")
      val (newS, duration) = time {
        runCommand(s, cmd, cmdOpts)
      }
      times += cmd.name -> duration
      newS
    }

    VCFReport.report()

    // Thread.sleep(60*60*1000)

    info(s"timing:\n${
      times.map { case (name, duration) =>
        s"  $name: ${formatTime(duration)}"
      }.mkString("\n")
    }")
  }

  def main(args: Array[String]) {

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
      AnnotateSamples,
      AnnotateVariants,
      Cache,
      ConvertAnnotations,
      Count,
      DownsampleVariants,
      ExportPlink,
      ExportGenotypes,
      ExportSamples,
      ExportVariants,
      ExportVCF,
      FilterGenotypes,
      FamSummary,
      FilterVariants,
      FilterSamples,
      GenDataset,
      Grep,
      GQByDP,
      GQHist,
      ImportVCF,
      LinearRegressionCommand,
      MendelErrorsCommand,
      SplitMulti,
      PCA,
      Read,
      RenameSamples,
      Repartition,
      SampleQC,
      ShowAnnotations,
      VariantQC,
      VEP,
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
        val visibleCommands = commands.filterNot(_.hidden)
        val maxLen = visibleCommands.map(_.name.length).max
        visibleCommands
          .foreach(cmd => println("  " + cmd.name + (" " * (maxLen - cmd.name.length + 2))
            + cmd.description))
        sys.exit(0)
      }
    } catch {
      case e: CmdLineException =>
        println(e.getMessage)
        sys.exit(1)
    }

    val logProps = new Properties()
    if (options.logQuiet) {
      logProps.put("log4j.rootLogger", "OFF, stderr")

      logProps.put("log4j.appender.stderr", "org.apache.log4j.ConsoleAppender")
      logProps.put("log4j.appender.stderr.Target", "System.err")
      logProps.put("log4j.appender.stderr.threshold", "OFF")
      logProps.put("log4j.appender.stderr.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.stderr.layout.ConversionPattern", "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    } else {
      logProps.put("log4j.rootLogger", "INFO, logfile")

      logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
      logProps.put("log4j.appender.logfile.append", options.logAppend.toString)
      logProps.put("log4j.appender.logfile.file", options.logFile)
      logProps.put("log4j.appender.logfile.threshold", "INFO")
      logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.logfile.layout.ConversionPattern", "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    }

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)

    if (splitArgs.length == 1)
      fatal("no commands given")

    val invocations: Array[(Command, Command#Options, Array[String])] = splitArgs.tail
      .map(args => {
        val cmdName = args(0)
        nameCommand.get(cmdName) match {
          case Some(cmd) =>
            (cmd, cmd.parseArgs(args.tail), args)
          case None =>
            fatal("unknown command `" + cmdName + "'")
        }
      })

    val conf = new SparkConf().setAppName("Hail")
    if (options.master != null)
      conf.setMaster(options.master)
    else if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    conf.set("spark.ui.showConsoleProgress", "false")

    conf.set("spark.sql.parquet.compression.codec", options.parquetCompression)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val sc = new SparkContext(conf)
    val progressBar = ProgressBarBuilder.build(sc)

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

    HailConfiguration.stacktrace = options.stacktrace
    HailConfiguration.installDir = new File(jar).getParent + "/.."
    HailConfiguration.tmpDir = options.tmpDir

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    runCommands(sc, sqlContext, invocations)

    sc.stop()
    progressBar.stop()
  }

}
