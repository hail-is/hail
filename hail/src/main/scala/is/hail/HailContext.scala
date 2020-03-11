package is.hail

import java.io.InputStream
import java.util.Properties

import is.hail.annotations._
import is.hail.backend.Backend
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.{BaseIR, ExecuteContext}
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual._
import is.hail.io.bgen.IndexBgen
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.io.index._
import is.hail.io.vcf._
import is.hail.io.{AbstractTypedCodecSpec, Decoder}
import is.hail.rvd.{AbstractIndexSpec, RVDContext}
import is.hail.sparkextras.{ContextRDD, IndexReadRDD}
import is.hail.utils.{log, _}
import is.hail.variant.ReferenceGenome
import org.apache.hadoop
import org.apache.log4j.{ConsoleAppender, LogManager, PatternLayout, PropertyConfigurator}
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.executor.InputMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.json4s.Extraction
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

case class FilePartition(index: Int, file: String) extends Partition

object HailContext {
  val tera: Long = 1024L * 1024L * 1024L * 1024L

  val logFormat: String = "%d{yyyy-MM-dd HH:mm:ss} %c{1}: %p: %m%n"

  private val contextLock = new Object()

  private var theContext: HailContext = _

  def isInitialized: Boolean = contextLock.synchronized {
    theContext != null
  }

  def get: HailContext = contextLock.synchronized {
    assert(TaskContext.get() == null, "HailContext not available on worker")
    assert(theContext != null, "HailContext not initialized")
    theContext
  }

  def backend: Backend = get.backend

  def getFlag(flag: String): String = get.flags.get(flag)

  def setFlag(flag: String, value: String): Unit = get.flags.set(flag, value)

  def sFS: FS = get.sFS

  def bcFS: Broadcast[FS] = get.bcFS

  def checkSparkCompatibility(jarVersion: String, sparkVersion: String): Unit = {
    def majorMinor(version: String): String = version.split("\\.", 3).take(2).mkString(".")

    if (majorMinor(jarVersion) != majorMinor(sparkVersion))
      fatal(s"This Hail JAR was compiled for Spark $jarVersion, cannot run with Spark $sparkVersion.\n" +
        s"  The major and minor versions must agree, though the patch version can differ.")
    else if (jarVersion != sparkVersion)
      warn(s"This Hail JAR was compiled for Spark $jarVersion, running with Spark $sparkVersion.\n" +
        s"  Compatibility is not guaranteed.")
  }

  def createSparkConf(appName: String, master: Option[String],
    local: String, blockSize: Long): SparkConf = {
    require(blockSize >= 0)
    checkSparkCompatibility(is.hail.HAIL_SPARK_VERSION, org.apache.spark.SPARK_VERSION)

    val conf = new SparkConf().setAppName(appName)

    master match {
      case Some(m) =>
        conf.setMaster(m)
      case None =>
        if (!conf.contains("spark.master"))
          conf.setMaster(local)
    }

    conf.set("spark.logConf", "true")
    conf.set("spark.ui.showConsoleProgress", "false")

    conf.set("spark.kryoserializer.buffer.max", "1g")
    conf.set("spark.driver.maxResultSize", "0")

    conf.set(
      "spark.hadoop.io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "is.hail.io.compress.BGzipCodec," +
        "is.hail.io.compress.BGzipCodecTbi," +
        "org.apache.hadoop.io.compress.GzipCodec")

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "is.hail.kryo.HailKryoRegistrator")

    conf.set("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", (blockSize * 1024L * 1024L).toString)

    // load additional Spark properties from HAIL_SPARK_PROPERTIES
    val hailSparkProperties = System.getenv("HAIL_SPARK_PROPERTIES")
    if (hailSparkProperties != null) {
      hailSparkProperties
        .split(",")
        .foreach { p =>
          p.split("=") match {
            case Array(k, v) =>
              log.info(s"set Spark property from HAIL_SPARK_PROPERTIES: $k=$v")
              conf.set(k, v)
            case _ =>
              warn(s"invalid key-value property pair in HAIL_SPARK_PROPERTIES: $p")
          }
        }
    }
    conf
  }

  def configureAndCreateSparkContext(appName: String, master: Option[String],
    local: String, blockSize: Long): SparkContext = {
    val sc = new SparkContext(createSparkConf(appName, master, local, blockSize))
    sc
  }

  def checkSparkConfiguration(sc: SparkContext) {
    val conf = sc.getConf

    val problems = new ArrayBuffer[String]

    val serializer = conf.getOption("spark.serializer")
    val kryoSerializer = "org.apache.spark.serializer.KryoSerializer"
    if (!serializer.contains(kryoSerializer))
      problems += s"Invalid configuration property spark.serializer: required $kryoSerializer.  " +
        s"Found: ${ serializer.getOrElse("empty parameter") }."

    if (!conf.getOption("spark.kryo.registrator").exists(_.split(",").contains("is.hail.kryo.HailKryoRegistrator")))
      problems += s"Invalid config parameter: spark.kryo.registrator must include is.hail.kryo.HailKryoRegistrator." +
        s"Found ${ conf.getOption("spark.kryo.registrator").getOrElse("empty parameter.") }"

    if (problems.nonEmpty)
      fatal(
        s"""Found problems with SparkContext configuration:
           |  ${ problems.mkString("\n  ") }""".stripMargin)
  }

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean) {
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
        if (major.toInt > 8)
          fatal(s"Hail requires Java 8, found $versionString")
      case _ =>
        fatal(s"Unknown JVM version string: $versionString")
    }
  }

  def hailCompressionCodecs: Array[String] = Array(
    "org.apache.hadoop.io.compress.DefaultCodec",
    "is.hail.io.compress.BGzipCodec",
    "is.hail.io.compress.BGzipCodecTbi",
    "org.apache.hadoop.io.compress.GzipCodec")

  /**
    * If a HailContext has already been initialized, this function returns it regardless of the
    * parameters with which it was initialized.
    *
    * Otherwise, it initializes and returns a new HailContext.
    */
  def getOrCreate(sc: SparkContext = null,
    appName: String = "Hail",
    master: Option[String] = None,
    local: String = "local[*]",
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    minBlockSize: Long = 1L,
    branchingFactor: Int = 50,
    tmpDir: String = "/tmp",
    optimizerIterations: Int = 3): HailContext = contextLock.synchronized {

    if (theContext != null) {
      val hc = theContext
      if (sc == null) {
        warn("Requested that Hail be initialized with a new SparkContext, but Hail " +
          "has already been initialized. Different configuration settings will be ignored.")
      }
      val paramsDiff = (Map(
        "tmpDir" -> Seq(tmpDir, hc.tmpDir),
        "branchingFactor" -> Seq(branchingFactor, hc.branchingFactor),
        "minBlockSize" -> Seq(minBlockSize, hc.sc.getConf.getLong("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", 0L) / 1024L / 1024L)
      ) ++ master.map(m => "master" -> Seq(m, hc.sc.master))).filter(_._2.areDistinct())
      val paramsDiffStr = paramsDiff.map { case (name, Seq(provided, existing)) =>
        s"Param: $name, Provided value: $provided, Existing value: $existing"
      }.mkString("\n")
      if (paramsDiff.nonEmpty) {
        warn("Found differences between requested and initialized parameters. Ignoring requested " +
          s"parameters.\n$paramsDiffStr")
      }

      hc
    } else {
      apply(sc, appName, master, local, logFile, quiet, append, minBlockSize, branchingFactor,
        tmpDir, optimizerIterations)
    }
  }

  def apply(sc: SparkContext = null,
    appName: String = "Hail",
    master: Option[String] = None,
    local: String = "local[*]",
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    minBlockSize: Long = 1L,
    branchingFactor: Int = 50,
    tmpDir: String = "/tmp",
    optimizerIterations: Int = 3): HailContext = contextLock.synchronized {
    require(theContext == null)
    checkJavaVersion()

    {
      import breeze.linalg._
      import breeze.linalg.operators.{BinaryRegistry, OpMulMatrix}

      implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
        DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
    }

    configureLogging(logFile, quiet, append)

    val sparkContext = if (sc == null)
      configureAndCreateSparkContext(appName, master, local, minBlockSize)
    else {
      checkSparkConfiguration(sc)
      sc
    }

    sparkContext.hadoopConfiguration.set("io.compression.codecs", hailCompressionCodecs.mkString(","))

    if (!quiet)
      ProgressBarBuilder.build(sparkContext)

    val hc = new HailContext(SparkBackend(sparkContext), new HadoopFS(new SerializableHadoopConfiguration(sparkContext.hadoopConfiguration)), logFile, tmpDir, branchingFactor, optimizerIterations)
    sparkContext.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))

    info(s"Running Hail version ${ hc.version }")
    theContext = hc

    // needs to be after `theContext` is set, since this creates broadcasts
    ReferenceGenome.addDefaultReferences()

    hc
  }

  def clear() {
    ReferenceGenome.reset()
    IRFunctionRegistry.clearUserFunctions()
    theContext = null
  }

  def startProgressBar(sc: SparkContext) {
    ProgressBarBuilder.build(sc)
  }

  def readRowsPartition(
    makeDec: (InputStream) => Decoder
  )(r: Region,
    in: InputStream,
    metrics: InputMetrics = null
  ): Iterator[RegionValue] =
    new Iterator[RegionValue] {
      private val region = r
      private val rv = RegionValue(region)

      private val trackedIn = new ByteTrackingInputStream(in)
      private val dec =
        try {
          makeDec(trackedIn)
        } catch {
          case e: Exception =>
            in.close()
            throw e
        }

      private var cont: Byte = dec.readByte()
      if (cont == 0)
        dec.close()

      // can't throw
      def hasNext: Boolean = cont != 0

      def next(): RegionValue = {
        // !hasNext => cont == 0 => dec has been closed
        if (!hasNext)
          throw new NoSuchElementException("next on empty iterator")

        try {
          rv.setOffset(dec.readRegionValue(region))
          cont = dec.readByte()
          if (metrics != null) {
            ExposedMetrics.incrementRecord(metrics)
            ExposedMetrics.incrementBytes(metrics, trackedIn.bytesReadAndClear())
          }

          if (cont == 0)
            dec.close()

          rv
        } catch {
          case e: Exception =>
            dec.close()
            throw e
        }
      }

      override def finalize(): Unit = {
        dec.close()
      }
    }

  def readRowsIndexedPartition(
    makeDec: (InputStream) => Decoder
  )(ctx: RVDContext,
    in: InputStream,
    idxr: IndexReader,
    offsetField: Option[String],
    bounds: Option[Interval],
    metrics: InputMetrics = null
  ): Iterator[RegionValue] =
    if (bounds.isEmpty) {
      idxr.close()
      HailContext.readRowsPartition(makeDec)(ctx.r, in, metrics)
    } else {
      new Iterator[RegionValue] {
        private val region = ctx.region
        private val rv = RegionValue(region)
        private val idx = idxr.queryByInterval(bounds.get).buffered

        private val trackedIn = new ByteTrackingInputStream(in)
        private val field = offsetField.map { f =>
          idxr.annotationType.asInstanceOf[TStruct].fieldIdx(f)
        }
        private val dec =
          try {
            if (idx.hasNext) {
              val dec = makeDec(trackedIn)
              val i = idx.head
              val off = field.map { j =>
                i.annotation.asInstanceOf[Row].getAs[Long](j)
              }.getOrElse(i.recordOffset)
              dec.seek(off)
              dec
            } else {
              in.close()
              null
            }
          } catch {
            case e: Exception =>
              idxr.close()
              in.close()
              throw e
          }

        private var cont: Byte = if (dec != null) dec.readByte() else 0
        if (cont == 0) {
          idxr.close()
          if (dec != null) dec.close()
        }

        def hasNext: Boolean = cont != 0 && idx.hasNext

        def next(): RegionValue = {
          if (!hasNext)
            throw new NoSuchElementException("next on empty iterator")

          try {
            idx.next()
            rv.setOffset(dec.readRegionValue(region))
            cont = dec.readByte()
            if (metrics != null) {
              ExposedMetrics.incrementRecord(metrics)
              ExposedMetrics.incrementBytes(metrics, trackedIn.bytesReadAndClear())
            }

            if (cont == 0) {
              dec.close()
              idxr.close()
            }

            rv
          } catch {
            case e: Exception =>
              dec.close()
              idxr.close()
              throw e
          }
        }

        override def finalize(): Unit = {
          idxr.close()
          if (dec != null) dec.close()
        }
      }
    }

  def readSplitRowsPartition(
    mkRowsDec: (InputStream) => Decoder,
    mkEntriesDec: (InputStream) => Decoder,
    mkInserter: (Int, Region) => (is.hail.asm4s.AsmFunction5[is.hail.annotations.Region,Long,Boolean,Long,Boolean,Long])
  )(ctx: RVDContext,
    isRows: InputStream,
    isEntries: InputStream,
    idxr: Option[IndexReader],
    rowsOffsetField: Option[String],
    entriesOffsetField: Option[String],
    bounds: Option[Interval],
    partIdx: Int,
    metrics: InputMetrics = null
  ): Iterator[RegionValue] = new Iterator[RegionValue] {
    private val region = ctx.region
    private val rv = RegionValue(region)
    private val idx = idxr.map(_.queryByInterval(bounds.get).buffered)

    private val trackedRowsIn = new ByteTrackingInputStream(isRows)
    private val trackedEntriesIn = new ByteTrackingInputStream(isEntries)

    private val rowsIdxField = rowsOffsetField.map { f => idxr.get.annotationType.asInstanceOf[TStruct].fieldIdx(f) }
    private val entriesIdxField = entriesOffsetField.map { f => idxr.get.annotationType.asInstanceOf[TStruct].fieldIdx(f) }

    private val inserter = mkInserter(partIdx, ctx.freshRegion)
    private val rows = try {
      if (idx.map(_.hasNext).getOrElse(true)) {
        val dec = mkRowsDec(trackedRowsIn)
        idx.map { idx =>
          val i = idx.head
          val off = rowsIdxField.map { j => i.annotation.asInstanceOf[Row].getAs[Long](j) }.getOrElse(i.recordOffset)
          dec.seek(off)
        }
        dec
      } else {
        isRows.close()
        isEntries.close()
        null
      }
    } catch {
      case e: Exception =>
        idxr.map(_.close())
        isRows.close()
        isEntries.close()
        throw e
    }
    private val entries = try {
      if (rows == null) {
        null
      } else {
        val dec = mkEntriesDec(trackedEntriesIn)
        idx.map { idx =>
          val i = idx.head
          val off = entriesIdxField.map { j => i.annotation.asInstanceOf[Row].getAs[Long](j) }.getOrElse(i.recordOffset)
          dec.seek(off)
        }
        dec
      }
    } catch {
      case e: Exception =>
        idxr.map(_.close())
        isRows.close()
        isEntries.close()
        throw e
    }

    require(!((rows == null) ^ (entries == null)))
    private def nextCont(): Byte = {
      val br = rows.readByte()
      val be = entries.readByte()
      assert(br == be)
      br
    }

    private var cont: Byte = if (rows != null) nextCont() else 0

    def hasNext: Boolean = cont != 0 && idx.map(_.hasNext).getOrElse(true)

    def next(): RegionValue = {
      if (!hasNext)
        throw new NoSuchElementException("next on empty iterator")

      try {
        idx.map(_.next())
        val rowOff = rows.readRegionValue(region)
        val entOff = entries.readRegionValue(region)
        val off = inserter(region, rowOff, false, entOff, false)
        rv.setOffset(off)
        cont = nextCont()

        if (cont == 0) {
          rows.close()
          entries.close()
          idxr.map(_.close())
        }

        rv
      } catch {
        case e: Exception =>
          rows.close()
          entries.close()
          idxr.map(_.close())
          throw e
      }
    }

    override def finalize(): Unit = {
      idxr.map(_.close())
      if (rows != null) rows.close()
      if (entries != null) entries.close()
    }
  }

  private[this] val codecsKey = "io.compression.codecs"
  private[this] val hadoopGzipCodec = "org.apache.hadoop.io.compress.GzipCodec"
  private[this] val hailGzipAsBGZipCodec = "is.hail.io.compress.BGzipCodecGZ"

  def maybeGZipAsBGZip[T](force: Boolean)(body: => T): T = {
    val fs = HailContext.get.sFS
    if (!force)
      body
    else {
      val defaultCodecs = fs.getProperty(codecsKey)
      fs.setProperty(codecsKey, defaultCodecs.replaceAllLiterally(hadoopGzipCodec, hailGzipAsBGZipCodec))
      try {
        body
      } finally {
        fs.setProperty(codecsKey, defaultCodecs)
      }
    }
  }

  def pyRemoveIrVector(id: Int) {
    get.irVectors.remove(id)
  }
}

class HailContext private(
  val backend: Backend,
  val sFS: FS,
  val logFile: String,
  val tmpDirPath: String,
  val branchingFactor: Int,
  val optimizerIterations: Int) {
  lazy val sc: SparkContext = backend.asSpark().sc

  lazy val sparkSession = SparkSession.builder().config(sc.getConf).getOrCreate()
  lazy val bcFS: Broadcast[FS] = sc.broadcast(sFS)

  val tmpDir = TempDir.createTempDir(tmpDirPath, sFS)
  info(s"Hail temporary directory: $tmpDir")

  val flags: HailFeatureFlags = new HailFeatureFlags()

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
    regex: String,
    files: Seq[String],
    maxLines: Int
  ): Map[String, Array[WithContext[String]]] = {
    val regexp = regex.r
    sc.textFilesLines(sFS.globAll(files))
      .filter(line => regexp.findFirstIn(line.value).isDefined)
      .take(maxLines)
      .groupBy(_.source.asInstanceOf[Context].file)
  }

  def grepPrint(regex: String, files: Seq[String], maxLines: Int) {
    fileAndLineCounts(regex, files, maxLines).foreach { case (file, lines) =>
      info(s"$file: ${ lines.length } ${ plural(lines.length, "match", "matches") }:")
      lines.map(_.value).foreach { line =>
        val (screen, logged) = line.truncatable().strings
        log.info("\t" + logged)
        println(s"\t$screen")
      }
    }
  }

  def grepReturn(regex: String, files: Seq[String], maxLines: Int): Array[(String, Array[String])] =
    fileAndLineCounts(regex, files, maxLines).mapValues(_.map(_.value)).toArray

  def getTemporaryFile(nChar: Int = 10, prefix: Option[String] = None, suffix: Option[String] = None): String =
    sFS.getTemporaryFile(tmpDir, nChar, prefix, suffix)

  def indexBgen(files: java.util.List[String],
    indexFileMap: java.util.Map[String, String],
    rg: Option[String],
    contigRecoding: java.util.Map[String, String],
    skipInvalidLoci: Boolean) {
    indexBgen(files.asScala, indexFileMap.asScala.toMap, rg, contigRecoding.asScala.toMap, skipInvalidLoci)
  }

  def indexBgen(files: Seq[String],
    indexFileMap: Map[String, String] = null,
    rg: Option[String] = None,
    contigRecoding: Map[String, String] = Map.empty[String, String],
    skipInvalidLoci: Boolean = false) {
    ExecuteContext.scoped { ctx =>
      IndexBgen(this, files.toArray, indexFileMap, rg, contigRecoding, skipInvalidLoci, ctx)
    }
    info(s"Number of BGEN files indexed: ${ files.length }")
  }

  def readPartitions[T: ClassTag](
    path: String,
    partFiles: Array[String],
    read: (Int, InputStream, InputMetrics) => Iterator[T],
    optPartitioner: Option[Partitioner] = None): RDD[T] = {
    val nPartitions = partFiles.length

    val localFS = bcFS

    new RDD[T](sc, Nil) {
      def getPartitions: Array[Partition] =
        Array.tabulate(nPartitions)(i => FilePartition(i, partFiles(i)))

      override def compute(split: Partition, context: TaskContext): Iterator[T] = {
        val p = split.asInstanceOf[FilePartition]
        val filename = path + "/parts/" + p.file
        val in = localFS.value.unsafeReader(filename)
        read(p.index, in, context.taskMetrics().inputMetrics)
      }

      @transient override val partitioner: Option[Partitioner] = optPartitioner
    }
  }

  def readIndexedPartitions(
    path: String,
    indexSpec: AbstractIndexSpec,
    partFiles: Array[String],
    intervalBounds: Option[Array[Interval]] = None
  ): RDD[(InputStream, IndexReader, Option[Interval], InputMetrics)] = {
    val idxPath = indexSpec.relPath
    val nPartitions = partFiles.length
    val localFS = bcFS
    val (keyType, annotationType) = indexSpec.types
    indexSpec.offsetField.foreach { f =>
      require(annotationType.asInstanceOf[TStruct].hasField(f))
      require(annotationType.asInstanceOf[TStruct].fieldType(f) == TInt64)
    }
    val (leafPType: PStruct, leafDec) = indexSpec.leafCodec.buildDecoder(indexSpec.leafCodec.encodedVirtualType)
    val (intPType: PStruct, intDec) = indexSpec.internalNodeCodec.buildDecoder(indexSpec.internalNodeCodec.encodedVirtualType)
    val mkIndexReader = IndexReaderBuilder.withDecoders(leafDec, intDec, keyType, annotationType, leafPType, intPType)

    new IndexReadRDD(sc, partFiles, intervalBounds, (p, context) => {
      val fs = localFS.value
      val idxname = s"$path/$idxPath/${ p.file }.idx"
      val filename = s"$path/parts/${ p.file }"
      val idxr = mkIndexReader(fs, idxname, 8) // default cache capacity
      val in = fs.unsafeReader(filename)
      (in, idxr, p.bounds, context.taskMetrics().inputMetrics)
    })
  }

  def readRows(
    path: String,
    enc: AbstractTypedCodecSpec,
    partFiles: Array[String],
    requestedType: TStruct
  ): (PStruct, ContextRDD[RegionValue]) = {
    val (pType: PStruct, makeDec) = enc.buildDecoder(requestedType)
    (pType, ContextRDD.weaken(readPartitions(path, partFiles, (_, is, m) => Iterator.single(is -> m)))
      .cmapPartitions { (ctx, it) =>
        assert(it.hasNext)
        val (is, m) = it.next
        assert(!it.hasNext)
        HailContext.readRowsPartition(makeDec)(ctx.r, is, m)
      })
  }

  def readIndexedRows(
    path: String,
    indexSpec: AbstractIndexSpec,
    enc: AbstractTypedCodecSpec,
    partFiles: Array[String],
    bounds: Array[Interval],
    requestedType: TStruct
  ): (PStruct, ContextRDD[RegionValue]) = {
    val (pType: PStruct, makeDec) = enc.buildDecoder(requestedType)
    (pType, ContextRDD.weaken(readIndexedPartitions(path, indexSpec, partFiles, Some(bounds)))
      .cmapPartitions { (ctx, it) =>
        assert(it.hasNext)
        val (is, idxr, bounds, m) = it.next
        assert(!it.hasNext)
        HailContext.readRowsIndexedPartition(makeDec)(ctx, is, idxr, indexSpec.offsetField, bounds, m)
      })
  }

  def readRowsSplit(
    ctx: ExecuteContext,
    pathRows: String,
    pathEntries: String,
    indexSpecRows: Option[AbstractIndexSpec],
    indexSpecEntries: Option[AbstractIndexSpec],
    rowsEnc: AbstractTypedCodecSpec,
    entriesEnc: AbstractTypedCodecSpec,
    partFiles: Array[String],
    bounds: Array[Interval],
    requestedTypeRows: TStruct,
    requestedTypeEntries: TStruct
  ): (PStruct, ContextRDD[RegionValue]) = {
    require(!(indexSpecRows.isEmpty ^ indexSpecEntries.isEmpty))
    val localFS = bcFS
    val (rowsType: PStruct, makeRowsDec) = rowsEnc.buildDecoder(requestedTypeRows)
    val (entriesType: PStruct, makeEntriesDec) = entriesEnc.buildDecoder(requestedTypeEntries)

    val inserterIR = ir.InsertFields(
      ir.Ref("left", requestedTypeRows),
      requestedTypeEntries.fieldNames.map(f =>
          f -> ir.GetField(ir.Ref("right", requestedTypeEntries), f)))

    val (t: PStruct, makeInserter) = ir.Compile[Long, Long, Long](ctx,
      "left", rowsType,
      "right", entriesType,
      inserterIR)

    val nPartitions = partFiles.length
    val mkIndexReader = indexSpecRows.map { indexSpec =>
      val idxPath = indexSpec.relPath
      val (keyType, annotationType) = indexSpec.types
      indexSpec.offsetField.foreach { f =>
        require(annotationType.asInstanceOf[TStruct].hasField(f))
        require(annotationType.asInstanceOf[TStruct].fieldType(f) == TInt64)
      }
      indexSpecEntries.get.offsetField.foreach { f =>
        require(annotationType.asInstanceOf[TStruct].hasField(f))
        require(annotationType.asInstanceOf[TStruct].fieldType(f) == TInt64)
      }
      IndexReaderBuilder.fromSpec(indexSpec)
    }

    val rdd = new IndexReadRDD(sc, partFiles, indexSpecRows.map(_ => bounds), (p, context) => {
      val fs = localFS.value
      val idxr = mkIndexReader.map { mk =>
        val idxname = s"$pathRows/${ indexSpecRows.get.relPath }/${ p.file }.idx"
        mk(fs, idxname, 8) // default cache capacity
      }
      val inRows = fs.unsafeReader(s"$pathRows/parts/${ p.file }")
      val inEntries = fs.unsafeReader(s"$pathEntries/parts/${ p.file }")
      (inRows, inEntries, idxr, p.bounds, context.taskMetrics().inputMetrics)
    })

    val rowsOffsetField = indexSpecRows.flatMap(_.offsetField)
    val entriesOffsetField = indexSpecEntries.flatMap(_.offsetField)
    (t, ContextRDD.weaken(rdd).cmapPartitionsWithIndex { (i, ctx, it) =>
      assert(it.hasNext)
      val (isRows, isEntries, idxr, bounds, m) = it.next
      assert(!it.hasNext)
      HailContext.readSplitRowsPartition(makeRowsDec, makeEntriesDec, makeInserter)(
        ctx, isRows, isEntries, idxr, rowsOffsetField, entriesOffsetField, bounds, i, m)
    })
  }

  def parseVCFMetadata(file: String): Map[String, Map[String, Map[String, String]]] = {
    LoadVCF.parseHeaderMetadata(this, Set.empty, TFloat64, file)
  }

  def pyParseVCFMetadataJSON(file: String): String = {
    val metadata = LoadVCF.parseHeaderMetadata(this, Set.empty, TFloat64, file)
    implicit val formats = defaultJSONFormats
    JsonMethods.compact(Extraction.decompose(metadata))
  }
}

class HailFeatureFlags {
  private[this] val flags: mutable.Map[String, String] =
    mutable.Map[String, String](
      "lower" -> sys.env.getOrElse("HAIL_DEV_LOWER", null),
      "lower_bm" -> sys.env.getOrElse("HAIL_DEV_LOWER_BM", null),
      "max_leader_scans" -> sys.env.getOrElse("HAIL_DEV_MAX_LEADER_SCANS", "1000"),
      "jvm_bytecode_dump" -> sys.env.getOrElse("HAIL_DEV_JVM_BYTECODE_DUMP", null),
      "use_packed_int_encoding" -> sys.env.getOrElse("HAIL_DEV_USE_PACKED_INT_ENCODING", null)
    )

  val available: java.util.ArrayList[String] =
    new java.util.ArrayList[String](java.util.Arrays.asList[String](flags.keys.toSeq: _*))

  def set(flag: String, value: String): Unit = {
    flags.update(flag, value)
  }

  def get(flag: String): String = flags(flag)

  def exists(flag: String): Boolean = flags.contains(flag)
}
