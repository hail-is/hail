package is.hail.backend.spark

import is.hail.annotations.UnsafeRow
import is.hail.asm4s._
import is.hail.expr.ir.IRParser
import is.hail.types.encoded.EType
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.HailContext
import is.hail.annotations.{Region, SafeRow}
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex, Validate}
import is.hail.expr.ir.lowering._
import is.hail.expr.ir._
import is.hail.types.physical.{PStruct, PTuple, PType}
import is.hail.types.virtual.{TStruct, TVoid, Type}
import is.hail.backend.{Backend, BackendContext, BroadcastValue, HailTaskContext, Py4JBackend}
import is.hail.io.fs.HadoopFS
import is.hail.utils._
import is.hail.io.bgen.IndexBgen
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}
import org.apache.spark.{ProgressBarBuilder, SparkConf, SparkContext, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import java.io.PrintWriter

import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.VCFsReader
import is.hail.linalg.{BlockMatrix, RowMatrix}
import is.hail.stats.LinearMixedModel
import is.hail.types.BlockMatrixType
import is.hail.variant.ReferenceGenome
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonAST.{JInt, JObject}


class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

class SparkTaskContext(ctx: TaskContext) extends HailTaskContext {
  type BackendType = SparkBackend
  override def stageId(): Int = ctx.stageId()
  override def partitionId(): Int = ctx.partitionId()
  override def attemptNumber(): Int = ctx.attemptNumber()
}

object SparkBackend {
  private var theSparkBackend: SparkBackend = _

  def sparkContext(op: String): SparkContext = HailContext.sparkBackend(op).sc

  def checkSparkCompatibility(jarVersion: String, sparkVersion: String): Unit = {
    def majorMinor(version: String): String = version.split("\\.", 3).take(2).mkString(".")

    if (majorMinor(jarVersion) != majorMinor(sparkVersion))
      fatal(s"This Hail JAR was compiled for Spark $jarVersion, cannot run with Spark $sparkVersion.\n" +
        s"  The major and minor versions must agree, though the patch version can differ.")
    else if (jarVersion != sparkVersion)
      warn(s"This Hail JAR was compiled for Spark $jarVersion, running with Spark $sparkVersion.\n" +
        s"  Compatibility is not guaranteed.")
  }

  def createSparkConf(appName: String, master: String,
    local: String, blockSize: Long): SparkConf = {
    require(blockSize >= 0)
    checkSparkCompatibility(is.hail.HAIL_SPARK_VERSION, org.apache.spark.SPARK_VERSION)

    val conf = new SparkConf().setAppName(appName)

    if (master != null)
      conf.setMaster(master)
    else {
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

  def configureAndCreateSparkContext(appName: String, master: String,
    local: String, blockSize: Long): SparkContext = {
    new SparkContext(createSparkConf(appName, master, local, blockSize))
  }

  def checkSparkConfiguration(sc: SparkContext) {
    val conf = sc.getConf

    val problems = new mutable.ArrayBuffer[String]

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

  def hailCompressionCodecs: Array[String] = Array(
    "org.apache.hadoop.io.compress.DefaultCodec",
    "is.hail.io.compress.BGzipCodec",
    "is.hail.io.compress.BGzipCodecTbi",
    "org.apache.hadoop.io.compress.GzipCodec")

  /**
    * If a SparkBackend has already been initialized, this function returns it regardless of the
    * parameters with which it was initialized.
    *
    * Otherwise, it initializes and returns a new HailContext.
    */
  def getOrCreate(sc: SparkContext = null,
    appName: String = "Hail",
    master: String = null,
    local: String = "local[*]",
    quiet: Boolean = false,
    minBlockSize: Long = 1L,
    tmpdir: String = "/tmp",
    localTmpdir: String = "file:///tmp"): SparkBackend = synchronized {
    if (theSparkBackend == null)
      return SparkBackend(sc, appName, master, local, quiet, minBlockSize, tmpdir, localTmpdir)

    // there should be only one SparkContext
    assert(sc == null || (sc eq theSparkBackend.sc))

    val initializedMinBlockSize =
      theSparkBackend.sc.getConf.getLong("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", 0L) / 1024L / 1024L
    if (minBlockSize != initializedMinBlockSize)
      warn(s"Requested minBlockSize $minBlockSize, but already initialized to $initializedMinBlockSize.  Ignoring requested setting.")

    if (master != null) {
      val initializedMaster = theSparkBackend.sc.master
      if (master != initializedMaster)
        warn(s"Requested master $master, but already initialized to $initializedMaster.  Ignoring requested setting.")
    }

    theSparkBackend
  }

  def apply(sc: SparkContext = null,
    appName: String = "Hail",
    master: String = null,
    local: String = "local[*]",
    quiet: Boolean = false,
    minBlockSize: Long = 1L,
    tmpdir: String,
    localTmpdir: String): SparkBackend = synchronized {
    require(theSparkBackend == null)

    var sc1 = sc
    if (sc1 == null)
      sc1 = configureAndCreateSparkContext(appName, master, local, minBlockSize)

    sc1.hadoopConfiguration.set("io.compression.codecs", hailCompressionCodecs.mkString(","))

    checkSparkConfiguration(sc1)

    if (!quiet)
      ProgressBarBuilder.build(sc1)

    sc1.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))

    theSparkBackend = new SparkBackend(tmpdir, localTmpdir, sc1)
    theSparkBackend
  }

  def stop(): Unit = synchronized {
    theSparkBackend = null
  }
}

class SparkBackend(
  val tmpdir: String,
  val localTmpdir: String,
  var sc: SparkContext
) extends Py4JBackend {
  lazy val sparkSession: SparkSession = SparkSession.builder().config(sc.getConf).getOrCreate()

  val fs: HadoopFS = new HadoopFS(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  val bmCache: SparkBlockMatrixCache = SparkBlockMatrixCache()

  def persistBlockMatrix(id: String, value: BlockMatrix, storageLevel: String): Unit = bmCache.persistBlockMatrix(id, value, storageLevel)

  def pyUnpersistBlockMatrix(id: String): Unit = bmCache.unpersistBlockMatrix(id)

  def getPersistedBlockMatrix(id: String): BlockMatrix = bmCache.getPersistedBlockMatrix(id)

  def getPersistedBlockMatrixType(id: String): BlockMatrixType = bmCache.getPersistedBlockMatrixType(id)

  def withExecuteContext[T]()(f: ExecuteContext => T): T = {
    ExecuteContext.scoped(tmpdir, localTmpdir, this, fs)(f)
  }

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new SparkBroadcastValue[T](sc.broadcast(value))

  def parallelizeAndComputeWithIndex(backendContext: BackendContext, collection: Array[Array[Byte]])(f: (Array[Byte], Int) => Array[Byte]): Array[Array[Byte]] = {
    val rdd = sc.parallelize(collection, numSlices = collection.length)
    rdd.mapPartitionsWithIndex { (i, it) =>
      HailTaskContext.setTaskContext(new SparkTaskContext(TaskContext.get))
      val elt = it.next()
      assert(!it.hasNext)
      Iterator.single(f(elt, i))
    }.collect()
  }

  def defaultParallelism: Int = sc.defaultParallelism

  override def asSpark(op: String): SparkBackend = this

  override def stop(): Unit = {
    sc.stop()
    sc = null
    super.stop()
    SparkBackend.stop()
  }

  def startProgressBar() {
    ProgressBarBuilder.build(sc)
  }

  private[this] def executionResultToAnnotation(ctx: ExecuteContext, result: Either[Unit, (PTuple, Long)]) = result match {
    case Left(x) => x
    case Right((pt, off)) => SafeRow(pt, off).get(0)
  }

  def jvmLowerAndExecute(
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None
  ): (Any, ExecutionTimer) = withExecuteContext() { ctx =>
    val (l, r) = _jvmLowerAndExecute(ctx, ir0, optimize, lowerTable, lowerBM, print)
    (executionResultToAnnotation(ctx, l), r)
  }

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None
  ): (Either[Unit, (PTuple, Long)], ExecutionTimer) = {
    val typesToLower: DArrayLowering.Type = (lowerTable, lowerBM) match {
      case (true, true) => DArrayLowering.All
      case (true, false) => DArrayLowering.TableOnly
      case (false, true) => DArrayLowering.BMOnly
      case (false, false) => throw new LowererUnsupportedOperation("no lowering enabled")
    }
    val ir = LoweringPipeline.darrayLowerer(true)(typesToLower).apply(ctx, ir0).asInstanceOf[IR]

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${ Pretty(ir) }")

    val res = ir.typ match {
      case TVoid =>
        val (_, f) = ctx.timer.time("Compile") {
          Compile[AsmFunction1RegionUnit](ctx,
            FastIndexedSeq[(String, PType)](),
            FastIndexedSeq(classInfo[Region]), UnitInfo,
            ir,
            print = print)
        }
        ctx.timer.time("Run")(Left(f(0, ctx.r)(ctx.r)))

      case _ =>
        val (pt: PTuple, f) = ctx.timer.time("Compile") {
          Compile[AsmFunction1RegionLong](ctx,
            FastIndexedSeq[(String, PType)](),
            FastIndexedSeq(classInfo[Region]), LongInfo,
            MakeTuple.ordered(FastSeq(ir)),
            print = print)
        }
        ctx.timer.time("Run")(Right((pt, f(0, ctx.r).apply(ctx.r))))
    }

    (res, ctx.timer)
  }

  def execute(ir: IR, optimize: Boolean): (Any, ExecutionTimer) =
    withExecuteContext() { ctx =>
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${ IRSize(ir) }")
      val (l, r) = _execute(ctx, ir, optimize)
      val javaObjResult = ctx.timer.time("convertRegionValueToAnnotation")(executionResultToAnnotation(ctx, l))
      log.info(s"finished execution of query $queryID")
      (javaObjResult, r)
    }

  private[this] def _execute(ctx: ExecuteContext, ir: IR, optimize: Boolean): (Either[Unit, (PTuple, Long)], ExecutionTimer) = {
    TypeCheck(ir)
    Validate(ir)
    try {
      val lowerTable = HailContext.getFlag("lower") != null
      val lowerBM = HailContext.getFlag("lower_bm") != null
      _jvmLowerAndExecute(ctx, ir, optimize, lowerTable, lowerBM)
    } catch {
      case e: LowererUnsupportedOperation if HailContext.getFlag("lower_only") != null => throw e
      case _: LowererUnsupportedOperation =>
        (CompileAndEvaluate._apply(ctx, ir, optimize = optimize), ctx.timer)
    }
  }

  def executeLiteral(id: Long): Long = {
    val ir = irMap(id).asInstanceOf[IR]
    val t = ir.typ
    assert(t.isRealizable)
    val (literalIR, timer) = withExecuteContext() { ctx =>
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID} of initial size ${ IRSize(ir) }")
      val (retVal, timer) = _execute(ctx, ir, true)
      val literalIR = retVal match {
        case Left(x) => throw new HailException("Can't create literal")
        case Right((pt, addr)) => GetFieldByIdx(EncodedLiteral.hailValueToByteArray(pt, addr, ctx), 0)
      }

      log.info(s"finished execution of query $queryID")
      (literalIR, timer)
    }
    timer.finish()
    timer.logInfo()
    addIR(literalIR)
  }

  def executeJSON(id: Long): String = {
    val ir = irMap(id).asInstanceOf[IR]
    val t = ir.typ
    val (value, timings) = execute(ir, optimize = true)
    val jsonValue = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    timings.finish()
    timings.logInfo()

    Serialization.write(Map("value" -> jsonValue, "timings" -> timings.asMap()))(new DefaultFormats {})
  }

  def encodeToBytes(id: Long, bufferSpecString: String): (String, Array[Byte]) = {
    val ir = irMap(id).asInstanceOf[IR]
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    withExecuteContext() { ctx =>
      _execute(ctx, ir, true)._1 match {
        case Left(_) => throw new RuntimeException("expression returned void")
        case Right((t, off)) =>
          assert(t.size == 1)
          val elementType = t.fields(0).typ
          val codec = TypedCodecSpec(
            EType.defaultFromPType(elementType), elementType.virtualType, bs)
          assert(t.isFieldDefined(off, 0))
          (elementType.toString, codec.encode(ctx, elementType, t.loadField(off, 0)))
      }
    }
  }

  def decodeToJSON(ptypeString: String, b: Array[Byte], bufferSpecString: String): String = {
    val t = IRParser.parsePType(ptypeString)
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    val codec = TypedCodecSpec(EType.defaultFromPType(t), t.virtualType, bs)
    withExecuteContext() { ctx =>
      val (pt, off) = codec.decode(ctx, t.virtualType, b, ctx.r)
      assert(pt.virtualType == t.virtualType)
      JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(
        UnsafeRow.read(pt, ctx.r, off), pt.virtualType))
    }
  }

  def pyIndexBgen(
    files: java.util.List[String],
    indexFileMap: java.util.Map[String, String],
    rg: String,
    contigRecoding: java.util.Map[String, String],
    skipInvalidLoci: Boolean) {
    withExecuteContext() { ctx =>
      IndexBgen(ctx, files.asScala.toArray, indexFileMap.asScala.toMap, Option(rg), contigRecoding.asScala.toMap, skipInvalidLoci)
    }
    info(s"Number of BGEN files indexed: ${ files.size() }")
  }

  def pyFromDF(df: DataFrame, jKey: java.util.List[String]): Long = {
    val key = jKey.asScala.toArray.toFastIndexedSeq
    val signature = SparkAnnotationImpex.importType(df.schema).setRequired(true).asInstanceOf[PStruct]
    withExecuteContext() { ctx =>
      addIR(TableLiteral(TableValue(ctx, signature.virtualType.asInstanceOf[TStruct], key, df.rdd, Some(signature))))
    }
  }

  def pyPersistMatrix(storageLevel: String, id: Long): Long = {
    val mir = irMap(id).asInstanceOf[MatrixIR]
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel: $storageLevel")
    }

    withExecuteContext() { ctx =>
      val tv = Interpret(mir, ctx, optimize = true)
      addIR(MatrixLiteral(mir.typ, TableLiteral(tv.persist(ctx, level))))
    }
  }

  def pyUnpersistMatrix(id: Long): Long = {
    val mir = irMap(id).asInstanceOf[MatrixIR]
    addIR(mir.unpersist())
  }

  def pyPersistTable(storageLevel: String, id: Long): Long = {
    val tir = irMap(id).asInstanceOf[TableIR]
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel: $storageLevel")
    }

    withExecuteContext() { ctx =>
      val tv = Interpret(tir, ctx, optimize = true)
      addIR(TableLiteral(tv.persist(ctx, level)))
    }
  }

  def pyUnpersistTable(id: Long): Long = {
    val tir = irMap(id).asInstanceOf[TableIR]
    addIR(tir.unpersist())
  }

  def pyToDF(id: Long): DataFrame = {
    val tir = irMap(id).asInstanceOf[TableIR]
    withExecuteContext() { ctx =>
      Interpret(tir, ctx).toDF()
    }
  }

  def pyImportVCFs(
    files: java.util.List[String],
    callFields: java.util.List[String],
    entryFloatTypeName: String,
    rg: String,
    contigRecoding: java.util.Map[String, String],
    arrayElementsRequired: Boolean,
    skipInvalidLoci: Boolean,
    gzAsBGZ: Boolean,
    forceGZ: Boolean,
    partitionsJSON: String,
    partitionsTypeStr: String,
    filter: String,
    find: String,
    replace: String,
    externalSampleIds: java.util.List[java.util.List[String]],
    externalHeader: String
  ): String = {
    withExecuteContext() { ctx =>
      val reader = new VCFsReader(ctx,
        files.asScala.toArray,
        callFields.asScala.toSet,
        entryFloatTypeName,
        Option(rg),
        Option(contigRecoding).map(_.asScala.toMap).getOrElse(Map.empty[String, String]),
        arrayElementsRequired,
        skipInvalidLoci,
        gzAsBGZ,
        forceGZ,
        TextInputFilterAndReplace(Option(find), Option(filter), Option(replace)),
        partitionsJSON, partitionsTypeStr,
        Option(externalSampleIds).map(_.asScala.map(_.asScala.toArray).toArray),
        Option(externalHeader))

      val irs = reader.read(ctx)
      val id = HailContext.get.addIrVector(irs)
      val out = JObject(
        "vector_ir_id" -> JInt(id),
        "length" -> JInt(irs.length),
        "type" -> reader.typ.toJSON)
      JsonMethods.compact(out)
    }
  }

  def pyReferenceAddLiftover(name: String, chainFile: String, destRGName: String): Unit = {
    withExecuteContext() { ctx =>
      ReferenceGenome.referenceAddLiftover(ctx, name, chainFile, destRGName)
    }
  }

  def pyFromFASTAFile(name: String, fastaFile: String, indexFile: String,
    xContigs: java.util.List[String], yContigs: java.util.List[String], mtContigs: java.util.List[String],
    parInput: java.util.List[String]): ReferenceGenome = {
    withExecuteContext() { ctx =>
      ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
        xContigs.asScala.toArray, yContigs.asScala.toArray, mtContigs.asScala.toArray, parInput.asScala.toArray)
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit = {
    withExecuteContext() { ctx =>
      ReferenceGenome.addSequence(ctx, name, fastaFile, indexFile)
    }
  }

  def pyExportBlockMatrix(
    pathIn: String, pathOut: String, delimiter: String, header: String, addIndex: Boolean, exportType: String,
    partitionSize: java.lang.Integer, entries: String): Unit = {
    withExecuteContext() { ctx =>
      val rm = RowMatrix.readBlockMatrix(fs, pathIn, partitionSize)
      entries match {
        case "full" =>
          rm.export(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "lower" =>
          rm.exportLowerTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "strict_lower" =>
          rm.exportStrictLowerTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "upper" =>
          rm.exportUpperTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "strict_upper" =>
          rm.exportStrictUpperTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
      }
    }
  }

  def pyFitLinearMixedModel(lmm: LinearMixedModel, pa_t: RowMatrix, a_t: RowMatrix): Long = {
    withExecuteContext() { ctx =>
      addIR(lmm.fit(ctx, pa_t, Option(a_t)))
    }
  }

  def lowerDistributedSort(ctx: ExecuteContext, stage: TableStage, sortFields: IndexedSeq[SortField], relationalLetsAbove: Map[String, IR]): TableStage = {
    // Use a local sort for the moment to enable larger pipelines to run
    LowerDistributedSort.localSort(ctx, stage, sortFields, relationalLetsAbove)
  }

  def pyImportFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): String =
    LoadPlink.importFamJSON(fs, path, isQuantPheno, delimiter, missingValue)
}
