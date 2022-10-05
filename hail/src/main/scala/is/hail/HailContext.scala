package is.hail

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.backend.{Backend, BroadcastValue, ExecuteContext}
import is.hail.expr.ir.BaseIR
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.io.fs.FS
import is.hail.io.index._
import is.hail.io.vcf._
import is.hail.io.{AbstractTypedCodecSpec, Decoder}
import is.hail.rvd.{AbstractIndexSpec, RVDContext}
import is.hail.sparkextras.{ContextRDD, IndexReadRDD}
import is.hail.types.physical.{PBaseStruct, PCanonicalTuple, PInt64Required, PStruct, PType}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.log4j.{ConsoleAppender, LogManager, PatternLayout, PropertyConfigurator}
import org.apache.spark._
import org.apache.spark.executor.InputMetrics
import org.apache.spark.rdd.RDD
import org.json4s.Extraction
import org.json4s.jackson.JsonMethods

import java.io.InputStream
import java.util.Properties
import scala.collection.mutable
import scala.reflect.ClassTag

case class FilePartition(index: Int, file: String) extends Partition

object HailContext {
  val tera: Long = 1024L * 1024L * 1024L * 1024L

  val logFormat: String = "%d{yyyy-MM-dd HH:mm:ss} %c{1}: %p: %m%n"

  private var theContext: HailContext = _

  def isInitialized: Boolean = synchronized {
    theContext != null
  }

  def get: HailContext = synchronized {
    assert(TaskContext.get() == null, "HailContext not available on worker")
    assert(theContext != null, "HailContext not initialized")
    theContext
  }

  def backend: Backend = get.backend

  def sparkBackend(op: String): SparkBackend = get.sparkBackend(op)

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean, skipLoggingConfiguration: Boolean): Unit = {
    if (!skipLoggingConfiguration) {
      val logProps = new Properties()

      logProps.put("log4j.rootLogger", "INFO, logfile")
      logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
      logProps.put("log4j.appender.logfile.append", append.toString)
      logProps.put("log4j.appender.logfile.file", logFile)
      logProps.put("log4j.appender.logfile.threshold", "INFO")
      logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.logfile.layout.ConversionPattern", HailContext.logFormat)

      LogManager.resetConfiguration()
      PropertyConfigurator.configure(logProps)
    }

    if (!quiet)
      consoleLog.addAppender(new ConsoleAppender(new PatternLayout(HailContext.logFormat), "System.err"))
  }

  def checkJavaVersion(): Unit = {
    val javaVersion = raw"(\d+)\.(\d+)\.(\d+).*".r
    val versionString = System.getProperty("java.version")
    versionString match {
      // old-style version: 1.MAJOR.MINOR
      // new-style version: MAJOR.MINOR.SECURITY (started in JRE 9)
      // see: https://docs.oracle.com/javase/9/migrate/toc.htm#JSMIG-GUID-3A71ECEF-5FC5-46FE-9BA9-88CBFCE828CB
      case javaVersion("1", major, minor) =>
        if (major.toInt < 8)
          fatal(s"Hail requires Java 1.8, found $versionString")
      case javaVersion(major, minor, security) =>
        if (major.toInt != 11)
          fatal(s"Hail requires Java 8 or 11, found $versionString")
      case _ =>
        fatal(s"Unknown JVM version string: $versionString")
    }
  }

  def getOrCreate(backend: Backend,
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    branchingFactor: Int = 50,
    skipLoggingConfiguration: Boolean = false,
    optimizerIterations: Int = 3): HailContext = {
    if (theContext == null)
      return HailContext(backend, logFile, quiet, append, branchingFactor, skipLoggingConfiguration, optimizerIterations)

    if (theContext.logFile != logFile)
      warn(s"Requested logFile $logFile, but already initialized to ${ theContext.logFile }.  Ignoring requested setting.")

    if (theContext.branchingFactor != branchingFactor)
      warn(s"Requested branchingFactor $branchingFactor, but already initialized to ${ theContext.branchingFactor }.  Ignoring requested setting.")

    if (theContext.optimizerIterations != optimizerIterations)
      warn(s"Requested optimizerIterations $optimizerIterations, but already initialized to ${ theContext.optimizerIterations }.  Ignoring requested setting.")

    theContext
  }

  def apply(backend: Backend,
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    branchingFactor: Int = 50,
    skipLoggingConfiguration: Boolean = false,
    optimizerIterations: Int = 3): HailContext = synchronized {
    require(theContext == null)
    checkJavaVersion()

    {
      import breeze.linalg._
      import breeze.linalg.operators.{BinaryRegistry, OpMulMatrix}

      implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
        DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
    }

    configureLogging(logFile, quiet, append, skipLoggingConfiguration)

    theContext = new HailContext(backend, logFile, branchingFactor, optimizerIterations)

    info(s"Running Hail version ${ theContext.version }")

    // needs to be after `theContext` is set, since this creates broadcasts
    ReferenceGenome.addDefaultReferences()

    theContext
  }

  def stop(): Unit = synchronized {
    ReferenceGenome.reset()
    IRFunctionRegistry.clearUserFunctions()
    backend.stop()

    theContext = null
  }

  def pyRemoveIrVector(id: Int) {
    get.irVectors.remove(id)
  }

  def readPartitions[T: ClassTag](
    fs: FS,
    path: String,
    partFiles: IndexedSeq[String],
    read: (Int, InputStream, InputMetrics) => Iterator[T],
    optPartitioner: Option[Partitioner] = None): RDD[T] = {
    val nPartitions = partFiles.length

    val fsBc = fs.broadcast

    new RDD[T](SparkBackend.sparkContext("readPartition"), Nil) {
      def getPartitions: Array[Partition] =
        Array.tabulate(nPartitions)(i => FilePartition(i, partFiles(i)))

      override def compute(split: Partition, context: TaskContext): Iterator[T] = {
        val p = split.asInstanceOf[FilePartition]
        val filename = path + "/parts/" + p.file
        val in = fsBc.value.open(filename)
        read(p.index, in, context.taskMetrics().inputMetrics)
      }

      @transient override val partitioner: Option[Partitioner] = optPartitioner
    }
  }
}

class HailContext private(
  var backend: Backend,
  val logFile: String,
  val branchingFactor: Int,
  val optimizerIterations: Int) {
  def stop(): Unit = HailContext.stop()

  def sparkBackend(op: String): SparkBackend = backend.asSpark(op)

  var checkRVDKeys: Boolean = false

  private var nextVectorId: Int = 0
  val irVectors: mutable.Map[Int, Array[_ <: BaseIR]] = mutable.Map.empty[Int, Array[_ <: BaseIR]]

  def addIrVector(irArray: Array[_ <: BaseIR]): Int = {
    val typ = irArray.head.typ
    irArray.foreach { ir =>
      if (ir.typ != typ)
        fatal("all ir vector items must have the same type")
    }
    irVectors(nextVectorId) = irArray
    nextVectorId += 1
    nextVectorId - 1
  }

  def version: String = is.hail.HAIL_PRETTY_VERSION

  private[this] def fileAndLineCounts(
    fs: FS,
    regex: String,
    files: Seq[String],
    maxLines: Int
  ): Map[String, Array[WithContext[String]]] = {
    val regexp = regex.r
    SparkBackend.sparkContext("fileAndLineCounts").textFilesLines(fs.globAll(files))
      .filter(line => regexp.findFirstIn(line.value).isDefined)
      .take(maxLines)
      .groupBy(_.source.file)
  }

  def grepPrint(fs: FS, regex: String, files: Seq[String], maxLines: Int) {
    fileAndLineCounts(fs, regex, files, maxLines).foreach { case (file, lines) =>
      info(s"$file: ${ lines.length } ${ plural(lines.length, "match", "matches") }:")
      lines.map(_.value).foreach { line =>
        val (screen, logged) = line.truncatable().strings
        log.info("\t" + logged)
        println(s"\t$screen")
      }
    }
  }

  def grepReturn(fs: FS, regex: String, files: Seq[String], maxLines: Int): Array[(String, Array[String])] =
    fileAndLineCounts(fs: FS, regex, files, maxLines).mapValues(_.map(_.value)).toArray

  def parseVCFMetadata(fs: FS, file: String): Map[String, Map[String, Map[String, String]]] = {
    LoadVCF.parseHeaderMetadata(fs, Set.empty, TFloat64, file)
  }

  def pyParseVCFMetadataJSON(fs: FS, file: String): String = {
    val metadata = LoadVCF.parseHeaderMetadata(fs, Set.empty, TFloat64, file)
    implicit val formats = defaultJSONFormats
    JsonMethods.compact(Extraction.decompose(metadata))
  }
}
