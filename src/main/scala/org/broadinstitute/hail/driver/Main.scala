package org.broadinstitute.hail.driver

import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.driver.Deduplicate.DuplicateReport
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.gen.GenReport
import org.broadinstitute.hail.io.vcf.VCFReport
import org.broadinstitute.hail.utils.FatalException
import org.kohsuke.args4j.{CmdLineException, CmdLineParser, Option => Args4jOption}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

object HailConfiguration {
  var tmpDir: String = "/tmp"

  var branchingFactor: Int = _

  def treeAggDepth(nPartitions: Int): Int = {
    (math.log(nPartitions) / math.log(branchingFactor) + 0.5).toInt.max(1)
  }
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

    @Args4jOption(name = "-b", aliases = Array("--min-block-size"), usage = "Minimum size of file splits in MB")
    var blockSize: Long = 1

    @Args4jOption(name = "-w", aliases = Array("--branching-factor"), usage = "Branching factor to use in tree aggregate")
    var branchingFactor: Int = 50

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

  def handleFatal(e: FatalException): Nothing = {
    log.error(s"hail: fatal: ${ e.logMsg }")
    System.err.println(s"hail: fatal: ${ e.msg }")
    sys.exit(1)
  }

  def handleFatal(cmd: Command, e: FatalException): Nothing = {
    log.error(s"hail: fatal: ${ cmd.name }: ${ e.logMsg }")
    System.err.println(s"hail: fatal: ${ cmd.name }: ${ e.msg }")

    sys.exit(1)
  }

  def expandException(e: Throwable): String = {
    s"${ e.getClass.getName }: ${ e.getLocalizedMessage }\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).map(exception => expandException(exception)).getOrElse("")
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
        val msg = s"hail: ${ cmd.name }: caught exception: ${ expandException(e) }"
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
    DuplicateReport.report()

    // Thread.sleep(60*60*1000)

    times += "total" -> times.map(_._2).sum

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

    {
      import breeze.linalg._
      import breeze.linalg.operators.{OpMulMatrix, BinaryRegistry}

      implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
        DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
    }

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

    val sc = configureAndCreateSparkContext("Hail", Option(options.master), local = "local[*]",
      parquetCompression = options.parquetCompression, blockSize = options.blockSize)
    configureLogging(logFile = options.logFile, quiet = options.logQuiet, append = options.logAppend)
    configureHail(branchingFactor = options.branchingFactor, tmpDir = options.tmpDir)

    val sqlContext = createSQLContext(sc)

    runCommands(sc, sqlContext, invocations)

    sc.stop()
  }
}
