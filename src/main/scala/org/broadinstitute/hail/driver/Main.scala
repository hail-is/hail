package org.broadinstitute.hail.driver

import java.io.File
import java.util.Properties

import org.apache.log4j.{LogManager, PropertyConfigurator}
import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.FatalException
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.gen.GenReport
import org.broadinstitute.hail.io.vcf.VCFReport
import org.kohsuke.args4j.{CmdLineException, CmdLineParser, Option => Args4jOption}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

object SparkManager {
  var _sc: SparkContext = _
  var _sqlContext: SQLContext = _

  def createSparkContext(appName: String, master: Option[String], local: String): SparkContext = {
    if (_sc == null) {
      val conf = new SparkConf().setAppName(appName)

      master match {
        case Some(m) =>
          conf.setMaster(m)
        case None =>
          if (!conf.contains("spark.master"))
            conf.setMaster(local)
      }

      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      _sc = new SparkContext(conf)
      _sc.hadoopConfiguration.set("io.compression.codecs",
        "org.apache.hadoop.io.compress.DefaultCodec,org.broadinstitute.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec")
    }

    _sc
  }

  def createSQLContext(): SQLContext = {
    assert(_sc != null)
    if (_sqlContext == null)
      _sqlContext = new org.apache.spark.sql.SQLContext(_sc)

    _sqlContext
  }
}

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

    @Args4jOption(name = "-b", aliases = Array("--blocksize"), usage = "Minimum size of file system splits")
    var blockSize: Int = 128

    @Args4jOption(required = false, name = "--parquet-compression", usage = "Parquet compression codec")
    var parquetCompression = "uncompressed"

    @Args4jOption(required = false, name = "-q", aliases = Array("--quiet"), usage = "Don't write log file")
    var logQuiet: Boolean = false

    @Args4jOption(required = false, name = "-t", aliases = Array("--tmpdir"), usage = "Temporary directory")
    var tmpDir: String = "/tmp"
  }

  private def fail(msg: String): Nothing = {
    log.error(msg)
    System.err.println(msg)
    sys.exit(1)
  }

  def handleFatal(e: Exception): Nothing = {
    val msg = s"hail: fatal: ${ e.getMessage }"
    fail(msg)
  }

  def handleFatal(cmd: Command, e: Exception): Nothing = {
    val msg = s"hail: fatal: ${ cmd.name }: ${ e.getMessage }"
    fail(msg)
  }

  def expandException(cmd: Command, e: Throwable): String = {
    s"${ e.getClass.getName }: ${ e.getLocalizedMessage }\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).map(exception => expandException(cmd, exception)).getOrElse("")
    }"
  }

  def handlePropagatedException(cmd: Command, e: Throwable) {
    e match {
      case f: FatalException => handleFatal(cmd, f)
      case _ => Option(e.getCause).foreach(c => handlePropagatedException(cmd, c))
    }
  }

  def runCommand(s: State, cmd: Command, cmdOpts: Command#Options): State = {
    try {
      cmd.runCommand(s, cmdOpts.asInstanceOf[cmd.Options])
    } catch {
      case e: Exception =>
        handlePropagatedException(cmd, e)
        val msg = s"hail: ${ cmd.name }: caught exception: ${ expandException(cmd, e) }"
        log.error(msg)
        System.err.println(msg)
        sys.exit(1)
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
    GenReport.report()

    // Thread.sleep(60*60*1000)

    info(s"timing:\n${
      times.map {
        case (name, duration) =>
          s"  $name: ${
            formatTime(duration)
          }"
      }.mkString("\n")
    }")
  }

  def main(args: Array[String]) {

    def splitBefore[T](a: Array[T], p: (T) => Boolean)
      (implicit tct: ClassTag[T]): Array[Array[T]] = {
      val r = mutable.ArrayBuilder.make[Array[T]]()
      val b = mutable.ArrayBuilder.make[T]()
      a.foreach {
        (x: T) =>
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

    val commandNames = ToplevelCommands.commandNames
    val splitArgs: Array[Array[String]] = splitBefore[String](args, (arg: String) => commandNames.contains(arg))

    val globalArgs = splitArgs(0)

    val options = new Options
    val parser = new CmdLineParser(options)
    try {
      parser.parseArgument((globalArgs: Iterable[String]).asJavaCollection)
      if (options.printUsage) {
        println("usage: hail [global options] <cmd1> [cmd1 args]")
        println("  [<cmd2> [cmd2 args] ... <cmdN> [cmdN args]]")
        println("")
        println("Global options:")
        new CmdLineParser(new Options).printUsage(System.out)
        println("")
        println("Commands:")
        ToplevelCommands.printCommands()
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
      fail(s"hail: fatal: no commands given")

    val invocations: Array[(Command, Command#Options, Array[String])] = splitArgs.tail
      .map {
        args =>
          val (cmd, cmdArgs) =
            try {
              ToplevelCommands.lookup(args)
            } catch {
              case e: FatalException =>
                handleFatal(e)
            }

          try {
            (cmd, cmd.parseArgs(cmdArgs): Command#Options, args)
          } catch {
            case e: FatalException =>
              handleFatal(cmd, e)
          }
      }

    val sc = SparkManager.createSparkContext("Hail", Option(options.master), "local[*]")

    val conf = sc.getConf
    conf.set("spark.ui.showConsoleProgress", "false")
    val progressBar = ProgressBarBuilder.build(sc)

    conf.set("spark.sql.parquet.compression.codec", options.parquetCompression)

    sc.hadoopConfiguration.setInt("mapreduce.input.fileinputformat.split.minsize", options.blockSize * 1024 * 1024)

    val sqlContext = SparkManager.createSQLContext()

    // FIXME separate entrypoints
    val jar = getClass.getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    sc.addJar(jar)

    HailConfiguration.installDir = new File(jar).getParent + "/.."
    HailConfiguration.tmpDir = options.tmpDir

    runCommands(sc, sqlContext, invocations)

    sc.stop()
    progressBar.stop()
  }
}
