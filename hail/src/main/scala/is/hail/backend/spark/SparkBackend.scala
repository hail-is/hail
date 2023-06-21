package is.hail.backend.spark

import cats.data.Kleisli
import cats.syntax.all._
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.ir.lowering.utils._
import is.hail.expr.ir.IRParser.parseType
import is.hail.expr.ir._
import is.hail.expr.ir.lowering._
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex, Validate}
import is.hail.io.fs._
import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.VCFsReader
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.linalg.{BlockMatrix, RowMatrix}
import is.hail.rvd.RVD
import is.hail.stats.LinearMixedModel
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.physical.{PStruct, PTuple}
import is.hail.types.virtual.{TArray, TInterval, TVoid}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import is.hail.{HailContext, HailFeatureFlags}
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s
import org.json4s.JsonAST.{JInt, JObject}
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{DefaultFormats, Formats}

import java.io.{Closeable, PrintWriter}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.higherKinds
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}


class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

object SparkTaskContext {
  def get(): SparkTaskContext = taskContext.get

  private[this] val taskContext: ThreadLocal[SparkTaskContext] = new ThreadLocal[SparkTaskContext]() {
    override def initialValue(): SparkTaskContext = {
      val sparkTC = TaskContext.get()
      assert(sparkTC != null, "Spark Task Context was null, maybe this ran on the driver?")
      sparkTC.addTaskCompletionListener[Unit] { (_: TaskContext) =>
        SparkTaskContext.finish()
      }

      // this must be the only place where SparkTaskContext classes are created
      new SparkTaskContext(sparkTC)
    }
  }

  def finish(): Unit = {
    taskContext.get().close()
    taskContext.remove()
  }
}


class SparkTaskContext private[spark](ctx: TaskContext) extends HailTaskContext {
  self =>
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
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    skipLoggingConfiguration: Boolean = false,
    minBlockSize: Long = 1L,
    tmpdir: String = "/tmp",
    localTmpdir: String = "file:///tmp",
    gcsRequesterPaysProject: String = null,
    gcsRequesterPaysBuckets: String = null
  ): SparkBackend = synchronized {
    if (theSparkBackend == null)
      return SparkBackend(sc, appName, master, local, logFile, quiet, append, skipLoggingConfiguration,
        minBlockSize, tmpdir, localTmpdir, gcsRequesterPaysProject, gcsRequesterPaysBuckets)

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
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    skipLoggingConfiguration: Boolean = false,
    minBlockSize: Long = 1L,
    tmpdir: String,
    localTmpdir: String,
    gcsRequesterPaysProject: String = null,
    gcsRequesterPaysBuckets: String = null
  ): SparkBackend = synchronized {
    require(theSparkBackend == null)

    if (!skipLoggingConfiguration)
      HailContext.configureLogging(logFile, quiet, append)

    var sc1 = sc
    if (sc1 == null)
      sc1 = configureAndCreateSparkContext(appName, master, local, minBlockSize)

    sc1.hadoopConfiguration.set("io.compression.codecs", hailCompressionCodecs.mkString(","))

    checkSparkConfiguration(sc1)

    if (!quiet)
      ProgressBarBuilder.build(sc1)

    sc1.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))

    theSparkBackend = new SparkBackend(tmpdir, localTmpdir, sc1, gcsRequesterPaysProject, gcsRequesterPaysBuckets)
    theSparkBackend
  }

  def stop(): Unit = synchronized {
    if (theSparkBackend != null) {
      theSparkBackend.sc.stop()
      theSparkBackend = null
      // Hadoop does not honor the hadoop configuration as a component of the cache key for file
      // systems, so we blow away the cache so that a new configuration can successfully take
      // effect.
      // https://github.com/hail-is/hail/pull/12133#issuecomment-1241322443
      hadoop.fs.FileSystem.closeAll()
    }
  }
}

// This indicates a narrow (non-shuffle) dependency on _rdd. It works since narrow dependency `getParents`
// is only used to compute preferred locations, which is something we don't need to worry about
class AnonymousDependency[T](val _rdd: RDD[T]) extends NarrowDependency[T](_rdd) {
  override def getParents(partitionId: Int): Seq[Int] = Seq.empty
}

class SparkBackend(
  val tmpdir: String,
  val localTmpdir: String,
  val sc: SparkContext,
  gcsRequesterPaysProject: String,
  gcsRequesterPaysBuckets: String
) extends Backend with Closeable with BackendWithCodeCache {
  assert(gcsRequesterPaysProject != null || gcsRequesterPaysBuckets == null)
  lazy val sparkSession: SparkSession = SparkSession.builder().config(sc.getConf).getOrCreate()
  private[this] val theHailClassLoader: HailClassLoader = new HailClassLoader(getClass().getClassLoader())

  override def canExecuteParallelTasksOnDriver: Boolean = false

  val fs: HadoopFS = {
    val conf = new Configuration(sc.hadoopConfiguration)
    if (gcsRequesterPaysProject != null) {
      if (gcsRequesterPaysBuckets == null) {
        conf.set("fs.gs.requester.pays.mode", "AUTO")
        conf.set("fs.gs.requester.pays.project.id", gcsRequesterPaysProject)
      } else {
        conf.set("fs.gs.requester.pays.mode", "CUSTOM")
        conf.set("fs.gs.requester.pays.project.id", gcsRequesterPaysProject)
        conf.set("fs.gs.requester.pays.buckets", gcsRequesterPaysBuckets)
      }
    }
    new HadoopFS(new SerializableHadoopConfiguration(conf))
  }
  private[this] val longLifeTempFileManager: TempFileManager = new OwningTempFileManager(fs)

  val bmCache: SparkBlockMatrixCache = SparkBlockMatrixCache()

  private[this] val flags = HailFeatureFlags.fromEnv()

  def getFlag(name: String): String = flags.get(name)

  def setFlag(name: String, value: String) = flags.set(name, value)

  val availableFlags: java.util.ArrayList[String] = flags.available

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit = bmCache.persistBlockMatrix(id, value, storageLevel)

  def unpersist(backendContext: BackendContext, id: String): Unit = unpersist(id)

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = bmCache.getPersistedBlockMatrix(id)

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = bmCache.getPersistedBlockMatrixType(id)

  def unpersist(id: String): Unit = bmCache.unpersistBlockMatrix(id)

  def createExecuteContextForTests(
    timer: ExecutionTimer,
    region: Region,
    selfContainedExecution: Boolean = true
  ): ExecuteContext = {
    val ctx = new ExecuteContext(
      tmpdir,
      localTmpdir,
      this,
      fs,
      region,
      timer,
      if (selfContainedExecution) null else new NonOwningTempFileManager(longLifeTempFileManager),
      theHailClassLoader,
      this.references,
      flags
    )

    ctx.backendContext = new BackendContext {
      override def executionCache: ExecutionCache =
        ExecutionCache.forTesting
    }

    ctx
  }

  def withExecuteContext[T](timer: ExecutionTimer, selfContainedExecution: Boolean = true)
                           (f: ExecuteContext => T): T =
    ExecuteContext.scoped(
      tmpdir,
      localTmpdir,
      this,
      fs,
      timer,
      if (selfContainedExecution) null else new NonOwningTempFileManager(longLifeTempFileManager),
      theHailClassLoader,
      this.references,
      flags
    ) { ctx =>
      ctx.backendContext = new BackendContext {
        override def executionCache: ExecutionCache =
          ExecutionCache.fromFlags(flags, fs, tmpdir)
      }
      f(ctx)
    }

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new SparkBroadcastValue[T](sc.broadcast(value))

  override def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    collection: Array[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None
  )(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte])
  : (Option[Throwable], IndexedSeq[(Int, Array[Byte])]) = {

    val sparkDeps =
      for {rvdDep <- dependency.toIndexedSeq.flatMap(_.deps)}
        yield new AnonymousDependency(rvdDep.asInstanceOf[RVDDependency].rvd.crdd.rdd)

    val rdd =
      new RDD[Try[(Int, Array[Byte])]](sc, sparkDeps) {
        override protected def getPartitions: Array[Partition] =
          for {(c, k) <- collection.zipWithIndex}
            yield SparkBackendComputeRDDPartition(c, k)

        override def compute(partition: Partition, context: TaskContext): Iterator[Try[(Int, Array[Byte])]] = {
          val sp = partition.asInstanceOf[SparkBackendComputeRDDPartition]
          val fs = new HadoopFS(null)
          Iterator.single(Try((sp.index, f(sp.data, SparkTaskContext.get(), theHailClassLoaderForSparkWorkers, fs))))
        }
      }

    val buffer = new ArrayBuffer[(Int, Array[Byte])](collection.length)
    var err = Option.empty[Throwable]

    rdd.collect().foreach {
      case Success((k, v)) => buffer.+=((k, v))
      case Failure(t) => err = err.orElse(Some(t))
    }

    (err, buffer)
  }

  def defaultParallelism: Int = sc.defaultParallelism

  override def asSpark(op: String): SparkBackend = this

  def stop(): Unit = SparkBackend.stop()

  def startProgressBar() {
    ProgressBarBuilder.build(sc)
  }

  private[this] def executionResultToAnnotation(result: Either[Unit, (PTuple, Long)]) = result match {
    case Left(x) => x
    case Right((pt, off)) => SafeRow(pt, off).get(0)
  }

  def jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None
  ): Any =
    _jvmLowerAndExecute[Lower](ir0, optimize, lowerTable, lowerBM, print)
      .map(executionResultToAnnotation)
      .runA(ctx, LoweringState())

  private[this] def _jvmLowerAndExecute[M[_]](
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None
  )(implicit M: MonadLower[M]): M[Either[Unit, (PTuple, Long)]] = {

    val typesToLower: DArrayLowering.Type = (lowerTable, lowerBM) match {
      case (true, true) => DArrayLowering.All
      case (true, false) => DArrayLowering.TableOnly
      case (false, true) => DArrayLowering.BMOnly
      case (false, false) => throw new LowererUnsupportedOperation("no lowering enabled")
    }

    LoweringPipeline.darrayLowerer(optimize)(typesToLower)(ir0).flatMap { case ir: IR =>
      if (!Compilable(ir))
        raisePretty(pretty => new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${pretty(ir)}"))
      else
        ir.typ match {
          case TVoid =>
            for {
              (_, f) <- timeM("Compile") {
                Compile[M, AsmFunction1RegionUnit](
                  FastIndexedSeq(),
                  FastIndexedSeq(classInfo[Region]), UnitInfo,
                  ir,
                  print = print
                )
              }

              _ <- timeM("Run") {
                scopedExecution { case (hcl, fs, htc, r) =>
                  M.pure(f(hcl, fs, htc, r)(r))
                }
              }

            } yield Left(())

          case _ =>
            for {
              (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) <- timeM("Compile") {
                Compile[M, AsmFunction1RegionLong](
                  FastIndexedSeq(),
                  FastIndexedSeq(classInfo[Region]), LongInfo,
                  MakeTuple.ordered(FastSeq(ir)),
                  print = print
                )
              }

              results <- timeM("Run") {
                scopedExecution { case (hcl, fs, htc, r) =>
                  M.pure(f(hcl, fs, htc, r)(r))
                }
              }
            } yield Right((pt, results))
        }
    }
  }

  def execute(timer: ExecutionTimer, ir: IR, optimize: Boolean): Any =
    withExecuteContext(timer) { ctx =>
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${ IRSize(ir) }")
      val l = _execute[Lower](ir, optimize).runA(ctx, LoweringState())
      val javaObjResult = ctx.timer.time("convertRegionValueToAnnotation")(executionResultToAnnotation(l))
      log.info(s"finished execution of query $queryID")
      javaObjResult
    }

  private[this] def _execute[M[_]](ir: IR, optimize: Boolean)
                                  (implicit M: MonadLower[M])
  : M[Either[Unit, (PTuple, Long)]] =
    TypeCheck(ir) *> {
      Validate(ir)
      val lowerTable = getFlag("lower") != null
      val lowerBM = getFlag("lower_bm") != null
      _jvmLowerAndExecute(ir, optimize, lowerTable, lowerBM)
        .handleErrorWith {
          case e: LowererUnsupportedOperation if getFlag("lower_only") != null => M.raiseError(e)
          case _: LowererUnsupportedOperation =>
            CompileAndEvaluate._apply(ir, optimize)
        }
    }

  def executeLiteral(ir: IR): IR = {
    val t = ir.typ
    assert(t.isRealizable)
    ExecutionTimer.logTime("SparkBackend.executeLiteral") { timer =>
      withExecuteContext(timer) { ctx =>
        val queryID = Backend.nextID()
        log.info(s"starting execution of query $queryID} of initial size ${ IRSize(ir) }")
        val literalIR = _execute[Lower](ir, optimize = true).runA(ctx, LoweringState()) match {
          case Left(_) => throw new HailException("Can't create literal")
          case Right((pt, addr)) => GetFieldByIdx(EncodedLiteral.fromPTypeAndAddress(pt, addr, ctx), 0)
        }
        log.info(s"finished execution of query $queryID")
        literalIR
      }
    }
  }

  def executeJSON(ir: IR): String = {
    val (jsonValue, timer) = ExecutionTimer.time("SparkBackend.executeJSON") { timer =>
      val t = ir.typ
      val value = execute(timer, ir, optimize = true)
      JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    }
    Serialization.write(Map("value" -> jsonValue, "timings" -> timer.toMap))(new DefaultFormats {})
  }

  def executeEncode(ir: IR, bufferSpecString: String, timed: Boolean): (Array[Byte], String) = {
    val (encodedValue, timer) = ExecutionTimer.time("SparkBackend.executeEncode") { timer =>
      withExecuteContext(timer) { ctx =>
        val queryID = Backend.nextID()
        log.info(s"starting execution of query $queryID of initial size ${ IRSize(ir) }")
        val res = _execute[Lower](ir, optimize = true).runA(ctx, LoweringState()) match {
          case Left(_) => Array[Byte]()
          case Right((t, off)) => encodeToBytes(ctx, t, off, bufferSpecString)
        }
        log.info(s"finished execution of query $queryID, result size is ${formatSpace(res.length)}")
        res
      }
    }
    val serializedTimer = if (timed) Serialization.write(Map("timings" -> timer.toMap))(new DefaultFormats {}) else ""
    (encodedValue, serializedTimer)
  }

  def encodeToBytes(ctx: ExecuteContext, t: PTuple, off: Long, bufferSpecString: String): Array[Byte] = {
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    assert(t.size == 1)
    val elementType = t.fields(0).typ
    val codec = TypedCodecSpec(
      EType.fromTypeAllOptional(elementType.virtualType), elementType.virtualType, bs)
    assert(t.isFieldDefined(off, 0))
    codec.encode(ctx, elementType, t.loadField(off, 0))
  }

  def decodeToJSON(ptypeString: String, b: Array[Byte], bufferSpecString: String): String = {
    ExecutionTimer.logTime("SparkBackend.decodeToJSON") { timer =>
      val t = IRParser.parsePType(ptypeString)
      val bs = BufferSpec.parseOrDefault(bufferSpecString)
      val codec = TypedCodecSpec(EType.defaultFromPType(t), t.virtualType, bs)
      withExecuteContext(timer) { ctx =>
        val (pt, off) = codec.decode(ctx, t.virtualType, b, ctx.r)
        assert(pt.virtualType == t.virtualType)
        JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(
          UnsafeRow.read(pt, ctx.r, off), pt.virtualType))
      }
    }
  }

  def pyFromDF(df: DataFrame, jKey: java.util.List[String]): TableIR =
    ExecutionTimer.logTime("SparkBackend.pyFromDF") { timer =>
      val key = jKey.asScala.toArray.toFastIndexedSeq
      val signature = SparkAnnotationImpex.importType(df.schema).setRequired(true).asInstanceOf[PStruct]
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        type F[A] = Kleisli[Try, ExecuteContext, A]
        (TableValue[F](signature.virtualType, key, df.rdd, Some(signature)) >>= (TableLiteral[F](_)))
          .run(ctx)
          .get
      }
    }

  def pyToDF(tir: TableIR): DataFrame =
    ExecutionTimer.logTime("SparkBackend.pyToDF") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        Interpret[Lower](tir).runA(ctx, LoweringState()).toDF()
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
    partitionsJSON: String,
    partitionsTypeStr: String,
    filter: String,
    find: String,
    replace: String,
    externalSampleIds: java.util.List[java.util.List[String]],
    externalHeader: String
  ): String = {
    ExecutionTimer.logTime("SparkBackend.pyImportVCFs") { timer =>
      withExecuteContext(timer) { ctx =>
        val reader = new VCFsReader(ctx,
          files.asScala.toArray,
          callFields.asScala.toSet,
          entryFloatTypeName,
          Option(rg),
          Option(contigRecoding).map(_.asScala.toMap).getOrElse(Map.empty[String, String]),
          arrayElementsRequired,
          skipInvalidLoci,
          TextInputFilterAndReplace(Option(find), Option(filter), Option(replace)),
          partitionsJSON, partitionsTypeStr,
          Option(externalSampleIds).map(_.asScala.map(_.asScala.toArray).toArray),
          Option(externalHeader))

        val irs = reader.read(ctx)
        val id = HailContext.get.addIrVector(irs)
        val out = JObject(
          "vector_ir_id" -> JInt(id),
          "length" -> JInt(irs.length),
          "type" -> reader.typ.pyJson)
        JsonMethods.compact(out)
      }
    }
  }

  def pyReadMultipleMatrixTables(jsonQuery: String): java.util.List[MatrixIR] = {
    log.info("pyReadMultipleMatrixTables: got query")
    val kvs = JsonMethods.parse(jsonQuery) match {
      case json4s.JObject(values) => values.toMap
    }

    val paths = kvs("paths").asInstanceOf[json4s.JArray].arr.toArray.map { case json4s.JString(s) => s }

    val intervalPointType = parseType(kvs("intervalPointType").asInstanceOf[json4s.JString].s)
    val intervalObjects = JSONAnnotationImpex.importAnnotation(kvs("intervals"), TArray(TInterval(intervalPointType)))
      .asInstanceOf[IndexedSeq[Interval]]

    val opts = NativeReaderOptions(intervalObjects, intervalPointType, filterIntervals = false)
    val matrixReaders: IndexedSeq[MatrixIR] = paths.map { p =>
      log.info(s"creating MatrixRead node for $p")
      val mnr = MatrixNativeReader(fs, p, Some(opts))
      MatrixRead(mnr.fullMatrixTypeWithoutUIDs, false, false, mnr): MatrixIR
    }
    log.info("pyReadMultipleMatrixTables: returning N matrix tables")
    matrixReaders.asJava
  }

  def pyLoadReferencesFromDataset(path: String): String = {
    val rgs = ReferenceGenome.fromHailDataset(fs, path)
    rgs.foreach(addReference)

    implicit val formats: Formats = defaultJSONFormats
    Serialization.write(rgs.map(_.toJSON).toFastIndexedSeq)
  }

  def pyAddReference(jsonConfig: String): Unit = addReference(ReferenceGenome.fromJSON(jsonConfig))
  def pyRemoveReference(name: String): Unit = removeReference(name)

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit = {
    ExecutionTimer.logTime("SparkBackend.pyReferenceAddLiftover") { timer =>
      withExecuteContext(timer) { ctx =>
        references(name).addLiftover(ctx, chainFile, destRGName)
      }
    }
  }
  def pyRemoveLiftover(name: String, destRGName: String) = references(name).removeLiftover(destRGName)

  def pyFromFASTAFile(name: String, fastaFile: String, indexFile: String,
    xContigs: java.util.List[String], yContigs: java.util.List[String], mtContigs: java.util.List[String],
    parInput: java.util.List[String]): String = {
    ExecutionTimer.logTime("SparkBackend.pyFromFASTAFile") { timer =>
      withExecuteContext(timer) { ctx =>
        val rg = ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
          xContigs.asScala.toArray, yContigs.asScala.toArray, mtContigs.asScala.toArray, parInput.asScala.toArray)
        rg.toJSONString
      }
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit = {
    ExecutionTimer.logTime("SparkBackend.pyAddSequence") { timer =>
      withExecuteContext(timer) { ctx =>
        references(name).addSequence(ctx, fastaFile, indexFile)
      }
    }
  }
  def pyRemoveSequence(name: String) = references(name).removeSequence()

  def pyExportBlockMatrix(
    pathIn: String, pathOut: String, delimiter: String, header: String, addIndex: Boolean, exportType: String,
    partitionSize: java.lang.Integer, entries: String): Unit = {
    ExecutionTimer.logTime("SparkBackend.pyExportBlockMatrix") { timer =>
      withExecuteContext(timer) { ctx =>
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
  }

  def pyFitLinearMixedModel(lmm: LinearMixedModel, pa_t: RowMatrix, a_t: RowMatrix): TableIR = {
    ExecutionTimer.logTime("SparkBackend.pyAddSequence") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        lmm.fit(ctx, pa_t, Option(a_t))
      }
    }
  }

  def parse_value_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): IR = {
    ExecutionTimer.logTime("SparkBackend.parse_value_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_value_ir(s, IRParserEnvironment(ctx, BindingEnv.eval(refMap.asScala.toMap.mapValues(IRParser.parseType).toSeq: _*), irMap.asScala.toMap))
      }
    }
  }

  def parse_table_ir(s: String, irMap: java.util.Map[String, BaseIR]): TableIR = {
    ExecutionTimer.logTime("SparkBackend.parse_table_ir") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = irMap.asScala.toMap))
      }
    }
  }

  def parse_matrix_ir(s: String, irMap: java.util.Map[String, BaseIR]): MatrixIR = {
    ExecutionTimer.logTime("SparkBackend.parse_matrix_ir") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, irMap = irMap.asScala.toMap))
      }
    }
  }

  def parse_blockmatrix_ir(
    s: String, irMap: java.util.Map[String, BaseIR]
  ): BlockMatrixIR = {
    ExecutionTimer.logTime("SparkBackend.parse_blockmatrix_ir") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, irMap = irMap.asScala.toMap))
      }
    }
  }

  override def lowerDistributedSort[M[_]](stage: TableStage, sortFields: IndexedSeq[SortField], rt: RTable)
                                         (implicit M: MonadLower[M]): M[TableReader] =
    if (getFlag("use_new_shuffle") != null)
      LowerDistributedSort.distributedSort(stage, sortFields, rt)
    else
      TableStageToRVD(stage).flatMap { case (globals, rvd) =>
        M.map2(M.ask, globals.toEncodedLiteral) { (ctx, globalsLit) =>

          if (sortFields.forall(_.sortOrder == Ascending))
            RVDTableReader(rvd.changeKey(ctx, sortFields.map(_.field)), globalsLit, rt)
          else {
            val rowType = rvd.rowType
            val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
              val i = rowType.fieldIdx(n)
              val f = rowType.fields(i)
              val fo = f.typ.ordering(ctx.stateManager)
              if (so == Ascending) fo else fo.reverse
            }.toArray

            val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

            val act = implicitly[ClassTag[Annotation]]

            val codec = TypedCodecSpec(rvd.rowPType, BufferSpec.wireSpec)
            val rdd = rvd.keyedEncodedRDD(ctx, codec, sortFields.map(_.field)).sortBy(_._1)(ord, act)
            val (rowPType: PStruct, orderedCRDD) = codec.decodeRDD(ctx, rowType, rdd.map(_._2))

            RVDTableReader(RVD.unkeyed(rowPType, orderedCRDD), globalsLit, rt)
          }
        }
      }

  def pyImportFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): String =
    LoadPlink.importFamJSON(fs, path, isQuantPheno, delimiter, missingValue)

  def close(): Unit = {
    longLifeTempFileManager.cleanup()
  }

  def tableToTableStage[M[_]](inputIR: TableIR, analyses: LoweringAnalyses)
                             (implicit M: MonadLower[M])
  : M[TableStage] =
    CanLowerEfficiently(inputIR).flatMap {
      case Left(failReason) =>
        log.info(s"SparkBackend: could not lower IR to table stage: $failReason")
        inputIR.analyzeAndExecute.flatMap(_.asTableStage)
      case Right(_) =>
        LowerTableIR.applyTable(inputIR, DArrayLowering.All, analyses)
    }
}

case class SparkBackendComputeRDDPartition(data: Array[Byte], index: Int) extends Partition

