package is.hail

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.{BaseIR, ExecuteContext}
import is.hail.io.fs.FS
import is.hail.io.index._
import is.hail.io.vcf._
import is.hail.io.{AbstractTypedCodecSpec, Decoder}
import is.hail.rvd.{AbstractIndexSpec, RVDContext}
import is.hail.sparkextras.{ContextRDD, IndexReadRDD}
import is.hail.types.physical.PStruct
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.log4j.{ConsoleAppender, LogManager, PatternLayout, PropertyConfigurator}
import org.apache.spark._
import org.apache.spark.executor.InputMetrics
import org.apache.spark.rdd.RDD
import org.json4s.Extraction
import org.json4s.JsonAST.{JArray, JObject, JString}
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

  def getFlag(flag: String): String = get.flags.get(flag)

  def setFlag(flag: String, value: String): Unit = get.flags.set(flag, value)

  def sparkBackend(op: String): SparkBackend = get.sparkBackend(op)

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean, skipLoggingConfiguration: Boolean) {
    if (!skipLoggingConfiguration) {
      val logProps = new Properties()

      logProps.put("log4j.rootLogger", "INFO, logfile")
      logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
      logProps.put("log4j.appender.logfile.append", append.toString)
      logProps.put("log4j.appender.logfile.file", logFile)
      logProps.put("log4j.appender.logfile.threshold", "INFO")
      logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.logfile.layout.ConversionPattern", HailContext.logFormat)

      logProps.put("log4j.logger.is.hail.services", "INFO, AlwaysOnConsoleAppender")
      logProps.put("log4j.logger.is.hail.backend.service", "INFO, AlwaysOnConsoleAppender")
      logProps.put("log4j.appender.AlwaysOnConsoleAppender", "org.apache.log4j.ConsoleAppender")
      logProps.put("log4j.appender.AlwaysOnConsoleAppender.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.AlwaysOnConsoleAppender.layout.ConversionPattern", HailContext.logFormat)

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

  def readRowsPartition(
    makeDec: (InputStream) => Decoder
  )(r: Region,
    in: InputStream,
    metrics: InputMetrics = null
  ): Iterator[Long] =
    new Iterator[Long] {
      private val region = r

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

      def next(): Long = {
        // !hasNext => cont == 0 => dec has been closed
        if (!hasNext)
          throw new NoSuchElementException("next on empty iterator")

        try {
          val res = dec.readRegionValue(region)
          cont = dec.readByte()
          if (metrics != null) {
            ExposedMetrics.incrementRecord(metrics)
            ExposedMetrics.incrementBytes(metrics, trackedIn.bytesReadAndClear())
          }

          if (cont == 0)
            dec.close()

          res
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
  ): Iterator[Long] =
    bounds match {
      case Some(b) =>
        new IndexReadIterator(makeDec, ctx.r, in, idxr, offsetField.orNull, b, metrics)
      case None =>
        idxr.close()
        HailContext.readRowsPartition(makeDec)(ctx.r, in, metrics)
    }

  def readSplitRowsPartition(fs: BroadcastValue[FS],
    mkRowsDec: (InputStream) => Decoder,
    mkEntriesDec: (InputStream) => Decoder,
    mkInserter: (FS, Int, Region) => AsmFunction3RegionLongLongLong
  )(ctx: RVDContext,
    isRows: InputStream,
    isEntries: InputStream,
    idxr: Option[IndexReader],
    rowsOffsetField: Option[String],
    entriesOffsetField: Option[String],
    bounds: Option[Interval],
    partIdx: Int,
    metrics: InputMetrics = null
  ): Iterator[Long] = new MaybeIndexedReadZippedIterator(mkRowsDec, mkEntriesDec, mkInserter(fs.value, partIdx, ctx.partitionRegion),
    ctx.r,
    isRows, isEntries,
    idxr.orNull, rowsOffsetField.orNull, entriesOffsetField.orNull, bounds.orNull, metrics)

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

  def readRows(
    ctx: ExecuteContext,
    path: String,
    enc: AbstractTypedCodecSpec,
    partFiles: Array[String],
    requestedType: TStruct
  ): (PStruct, ContextRDD[Long]) = {
    val fs = ctx.fs
    val (pType: PStruct, makeDec) = enc.buildDecoder(ctx, requestedType)
    (pType, ContextRDD.weaken(HailContext.readPartitions(fs, path, partFiles, (_, is, m) => Iterator.single(is -> m)))
      .cmapPartitions { (ctx, it) =>
        assert(it.hasNext)
        val (is, m) = it.next
        assert(!it.hasNext)
        HailContext.readRowsPartition(makeDec)(ctx.r, is, m)
      })
  }

  def readIndexedRows(
    ctx: ExecuteContext,
    path: String,
    indexSpec: AbstractIndexSpec,
    enc: AbstractTypedCodecSpec,
    partFiles: Array[String],
    bounds: Array[Interval],
    requestedType: TStruct
  ): (PStruct, ContextRDD[Long]) = {
    val (pType: PStruct, makeDec) = enc.buildDecoder(ctx, requestedType)
    (pType, ContextRDD.weaken(readIndexedPartitions(ctx, path, indexSpec, partFiles, Some(bounds)))
      .cmapPartitions { (ctx, it) =>
        assert(it.hasNext)
        val (is, idxr, bounds, m) = it.next
        assert(!it.hasNext)
        readRowsIndexedPartition(makeDec)(ctx, is, idxr, indexSpec.offsetField, bounds, m)
      })
  }

  def readIndexedPartitions(
    ctx: ExecuteContext,
    path: String,
    indexSpec: AbstractIndexSpec,
    partFiles: Array[String],
    intervalBounds: Option[Array[Interval]] = None
  ): RDD[(InputStream, IndexReader, Option[Interval], InputMetrics)] = {
    val idxPath = indexSpec.relPath
    val fsBc = ctx.fsBc
    val (keyType, annotationType) = indexSpec.types
    indexSpec.offsetField.foreach { f =>
      require(annotationType.asInstanceOf[TStruct].hasField(f))
      require(annotationType.asInstanceOf[TStruct].fieldType(f) == TInt64)
    }
    val (leafPType: PStruct, leafDec) = indexSpec.leafCodec.buildDecoder(ctx, indexSpec.leafCodec.encodedVirtualType)
    val (intPType: PStruct, intDec) = indexSpec.internalNodeCodec.buildDecoder(ctx, indexSpec.internalNodeCodec.encodedVirtualType)
    val mkIndexReader = IndexReaderBuilder.withDecoders(leafDec, intDec, keyType, annotationType, leafPType, intPType)

    new IndexReadRDD(partFiles, intervalBounds, { (p, context) =>
      val fs = fsBc.value
      val idxname = s"$path/$idxPath/${ p.file }.idx"
      val filename = s"$path/parts/${ p.file }"
      val idxr = mkIndexReader(fs, idxname, 8, SparkTaskContext.get().getRegionPool()) // default cache capacity
      val in = fs.open(filename)
      (in, idxr, p.bounds, context.taskMetrics().inputMetrics)
    })
  }


  def readRowsSplit(
    ctx: ExecuteContext,
    pathRows: String,
    pathEntries: String,
    indexSpecRows: Option[AbstractIndexSpec],
    indexSpecEntries: Option[AbstractIndexSpec],
    partFiles: Array[String],
    bounds: Array[Interval],
    makeRowsDec: InputStream => Decoder,
    makeEntriesDec: InputStream => Decoder,
    makeInserter: (FS, Int, Region) => AsmFunction3RegionLongLongLong
  ): ContextRDD[Long] = {
    require(!(indexSpecRows.isEmpty ^ indexSpecEntries.isEmpty))
    val fsBc = ctx.fsBc

    val mkIndexReader = indexSpecRows.map { indexSpec =>
      val (keyType, annotationType) = indexSpec.types
      indexSpec.offsetField.foreach { f =>
        require(annotationType.asInstanceOf[TStruct].hasField(f))
        require(annotationType.asInstanceOf[TStruct].fieldType(f) == TInt64)
      }
      indexSpecEntries.get.offsetField.foreach { f =>
        require(annotationType.asInstanceOf[TStruct].hasField(f))
        require(annotationType.asInstanceOf[TStruct].fieldType(f) == TInt64)
      }
      IndexReaderBuilder.fromSpec(ctx, indexSpec)
    }

    val rdd = new IndexReadRDD(partFiles, indexSpecRows.map(_ => bounds), (p, context) => {
      val fs = fsBc.value
      val idxr = mkIndexReader.map { mk =>
        val idxname = s"$pathRows/${ indexSpecRows.get.relPath }/${ p.file }.idx"
        mk(fs, idxname, 8, SparkTaskContext.get().getRegionPool()) // default cache capacity
      }
      val inRows = fs.open(s"$pathRows/parts/${ p.file }")
      val inEntries = fs.open(s"$pathEntries/parts/${ p.file }")
      (inRows, inEntries, idxr, p.bounds, context.taskMetrics().inputMetrics)
    })

    val rowsOffsetField = indexSpecRows.flatMap(_.offsetField)
    val entriesOffsetField = indexSpecEntries.flatMap(_.offsetField)
    ContextRDD.weaken(rdd).cmapPartitionsWithIndex { (i, ctx, it) =>
      assert(it.hasNext)
      val (isRows, isEntries, idxr, bounds, m) = it.next
      assert(!it.hasNext)
      HailContext.readSplitRowsPartition(fsBc, makeRowsDec, makeEntriesDec, makeInserter)(
        ctx, isRows, isEntries, idxr, rowsOffsetField, entriesOffsetField, bounds, i, m)
    }

  }
}

class HailContext private(
  val backend: Backend,
  val logFile: String,
  val branchingFactor: Int,
  val optimizerIterations: Int) {
  def stop(): Unit = HailContext.stop()

  def sparkBackend(op: String): SparkBackend = backend.asSpark(op)

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

object HailFeatureFlags {
  val defaults: Map[String, (String, String)] = Map[String, (String, String)](
    ("lower", ("HAIL_DEV_LOWER" -> null)),
    ("lower_only", ("HAIL_DEV_LOWER_ONLY" -> null)),
    ("lower_bm", ("HAIL_DEV_LOWER_BM" -> null)),
    ("max_leader_scans", ("HAIL_DEV_MAX_LEADER_SCANS" -> "1000")),
    ("distributed_scan_comb_op", ("HAIL_DEV_DISTRIBUTED_SCAN_COMB_OP" -> null)),
    ("jvm_bytecode_dump", ("HAIL_DEV_JVM_BYTECODE_DUMP" -> null)),
    ("use_packed_int_encoding", ("HAIL_DEV_USE_PACKED_INT_ENCODING" -> null)),
    ("use_column_encoding", ("HAIL_DEV_USE_COLUMN_ENCODING" -> null)),
    ("use_spicy_ptypes", ("HAIL_USE_SPICY_PTYPES" -> null)),
    ("log_service_timing", ("HAIL_DEV_LOG_SERVICE_TIMING" -> null)),
    ("cache_service_input", ("HAIL_DEV_CACHE_SERVICE_INPUT" -> null)),
    ("write_ir_files", ("HAIL_WRITE_IR_FILES" -> null)),
    ("method_split_ir_limit", ("HAIL_DEV_METHOD_SPLIT_LIMIT" -> "16"))
  )
}

class HailFeatureFlags {
  private[this] val flags: mutable.Map[String, String] =
    mutable.Map[String, String](HailFeatureFlags.defaults.mapValues { case (env, default) =>
      sys.env.getOrElse(env, default)
    }.toFastSeq: _*)

  val available: java.util.ArrayList[String] =
    new java.util.ArrayList[String](java.util.Arrays.asList[String](flags.keys.toSeq: _*))

  def set(flag: String, value: String): Unit = {
    flags.update(flag, value)
  }

  def get(flag: String): String = flags(flag)

  def exists(flag: String): Boolean = flags.contains(flag)

  def toJSONEnv: JArray =
    JArray(flags.filter { case (_, v) =>
      v != null
    }.map{ case (name, v) =>
      JObject(
        "name" -> JString(HailFeatureFlags.defaults(name)._1),
        "value" -> JString(v))
    }.toList)
}
