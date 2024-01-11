package is.hail.backend.spark

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex, Validate}
import is.hail.expr.ir.{IRParser, _}
import is.hail.expr.ir.IRParser.parseType
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs._
import is.hail.linalg.{BlockMatrix, RowMatrix}
import is.hail.rvd.RVD
import is.hail.stats.LinearMixedModel
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical.{PStruct, PTuple}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual.{TArray, TInterval, TStruct, TVoid}
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import java.io.{Closeable, PrintWriter}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}
import scala.util.control.NonFatal

import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s

class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

object SparkTaskContext {
  def get(): SparkTaskContext = taskContext.get

  private[this] val taskContext: ThreadLocal[SparkTaskContext] =
    new ThreadLocal[SparkTaskContext]() {
      override def initialValue(): SparkTaskContext = {
        val sparkTC = TaskContext.get()
        assert(sparkTC != null, "Spark Task Context was null, maybe this ran on the driver?")
        sparkTC.addTaskCompletionListener[Unit]((_: TaskContext) => SparkTaskContext.finish())

        // this must be the only place where SparkTaskContext classes are created
        new SparkTaskContext(sparkTC)
      }
    }

  def finish(): Unit = {
    taskContext.get().close()
    taskContext.remove()
  }
}

class SparkTaskContext private[spark] (ctx: TaskContext) extends HailTaskContext {
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
      fatal(
        s"This Hail JAR was compiled for Spark $jarVersion, cannot run with Spark $sparkVersion.\n" +
          s"  The major and minor versions must agree, though the patch version can differ."
      )
    else if (jarVersion != sparkVersion)
      warn(
        s"This Hail JAR was compiled for Spark $jarVersion, running with Spark $sparkVersion.\n" +
          s"  Compatibility is not guaranteed."
      )
  }

  def createSparkConf(appName: String, master: String, local: String, blockSize: Long)
    : SparkConf = {
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
        "org.apache.hadoop.io.compress.GzipCodec",
    )

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "is.hail.kryo.HailKryoRegistrator")

    conf.set(
      "spark.hadoop.mapreduce.input.fileinputformat.split.minsize",
      (blockSize * 1024L * 1024L).toString,
    )

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

  def configureAndCreateSparkContext(
    appName: String,
    master: String,
    local: String,
    blockSize: Long,
  ): SparkContext =
    new SparkContext(createSparkConf(appName, master, local, blockSize))

  def checkSparkConfiguration(sc: SparkContext): Unit = {
    val conf = sc.getConf

    val problems = new mutable.ArrayBuffer[String]

    val serializer = conf.getOption("spark.serializer")
    val kryoSerializer = "org.apache.spark.serializer.KryoSerializer"
    if (!serializer.contains(kryoSerializer))
      problems += s"Invalid configuration property spark.serializer: required $kryoSerializer.  " +
        s"Found: ${serializer.getOrElse("empty parameter")}."

    if (
      !conf.getOption("spark.kryo.registrator").exists(
        _.split(",").contains("is.hail.kryo.HailKryoRegistrator")
      )
    )
      problems += s"Invalid config parameter: spark.kryo.registrator must include is.hail.kryo.HailKryoRegistrator." +
        s"Found ${conf.getOption("spark.kryo.registrator").getOrElse("empty parameter.")}"

    if (problems.nonEmpty)
      fatal(
        s"""Found problems with SparkContext configuration:
           |  ${problems.mkString("\n  ")}""".stripMargin
      )
  }

  def hailCompressionCodecs: Array[String] = Array(
    "org.apache.hadoop.io.compress.DefaultCodec",
    "is.hail.io.compress.BGzipCodec",
    "is.hail.io.compress.BGzipCodecTbi",
    "org.apache.hadoop.io.compress.GzipCodec",
  )

  /** If a SparkBackend has already been initialized, this function returns it regardless of the
    * parameters with which it was initialized.
    *
    * Otherwise, it initializes and returns a new HailContext.
    */
  def getOrCreate(
    sc: SparkContext = null,
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
    gcsRequesterPaysBuckets: String = null,
  ): SparkBackend = synchronized {
    if (theSparkBackend == null)
      return SparkBackend(sc, appName, master, local, logFile, quiet, append,
        skipLoggingConfiguration,
        minBlockSize, tmpdir, localTmpdir, gcsRequesterPaysProject, gcsRequesterPaysBuckets)

    // there should be only one SparkContext
    assert(sc == null || (sc eq theSparkBackend.sc))

    val initializedMinBlockSize =
      theSparkBackend.sc.getConf.getLong(
        "spark.hadoop.mapreduce.input.fileinputformat.split.minsize",
        0L,
      ) / 1024L / 1024L
    if (minBlockSize != initializedMinBlockSize)
      warn(
        s"Requested minBlockSize $minBlockSize, but already initialized to $initializedMinBlockSize.  Ignoring requested setting."
      )

    if (master != null) {
      val initializedMaster = theSparkBackend.sc.master
      if (master != initializedMaster)
        warn(
          s"Requested master $master, but already initialized to $initializedMaster.  Ignoring requested setting."
        )
    }

    theSparkBackend
  }

  def apply(
    sc: SparkContext = null,
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
    gcsRequesterPaysBuckets: String = null,
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

    theSparkBackend =
      new SparkBackend(tmpdir, localTmpdir, sc1, gcsRequesterPaysProject, gcsRequesterPaysBuckets)
    theSparkBackend.addDefaultReferences()
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
  gcsRequesterPaysBuckets: String,
) extends Backend with Closeable with BackendWithCodeCache {
  assert(gcsRequesterPaysProject != null || gcsRequesterPaysBuckets == null)
  lazy val sparkSession: SparkSession = SparkSession.builder().config(sc.getConf).getOrCreate()

  private[this] val theHailClassLoader: HailClassLoader =
    new HailClassLoader(getClass().getClassLoader())

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

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String)
    : Unit = bmCache.persistBlockMatrix(id, value, storageLevel)

  def unpersist(backendContext: BackendContext, id: String): Unit = unpersist(id)

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix =
    bmCache.getPersistedBlockMatrix(id)

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType =
    bmCache.getPersistedBlockMatrixType(id)

  def unpersist(id: String): Unit = bmCache.unpersistBlockMatrix(id)

  def createExecuteContextForTests(
    timer: ExecutionTimer,
    region: Region,
    selfContainedExecution: Boolean = true,
  ): ExecuteContext =
    new ExecuteContext(
      tmpdir,
      localTmpdir,
      this,
      fs,
      region,
      timer,
      if (selfContainedExecution) null else new NonOwningTempFileManager(longLifeTempFileManager),
      theHailClassLoader,
      flags,
      new BackendContext {
        override val executionCache: ExecutionCache =
          ExecutionCache.forTesting
      },
      IrMetadata(None),
    )

  def withExecuteContext[T](timer: ExecutionTimer, selfContainedExecution: Boolean = true)
    : (ExecuteContext => T) => T =
    ExecuteContext.scoped(
      tmpdir,
      localTmpdir,
      this,
      fs,
      timer,
      if (selfContainedExecution) null else new NonOwningTempFileManager(longLifeTempFileManager),
      theHailClassLoader,
      flags,
      new BackendContext {
        override val executionCache: ExecutionCache =
          ExecutionCache.fromFlags(flags, fs, tmpdir)
      },
    )

  override def withExecuteContext[T](methodName: String)(f: ExecuteContext => T): T =
    ExecutionTimer.logTime(methodName) { timer =>
      ExecuteContext.scoped(
        tmpdir,
        tmpdir,
        this,
        fs,
        timer,
        null,
        theHailClassLoader,
        flags,
        new BackendContext {
          override val executionCache: ExecutionCache =
            ExecutionCache.fromFlags(flags, fs, tmpdir)
        },
      )(f)
    }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] =
    new SparkBroadcastValue[T](sc.broadcast(value))

  override def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    collection: Array[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None,
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): Array[Array[Byte]] = {
    val sparkDeps = dependency.toIndexedSeq
      .flatMap(dep =>
        dep.deps.map(rvdDep =>
          new AnonymousDependency(rvdDep.asInstanceOf[RVDDependency].rvd.crdd.rdd)
        )
      )

    new SparkBackendComputeRDD(sc, collection, f, sparkDeps).collect()
  }

  override def parallelizeAndComputeWithIndexReturnAllErrors(
    backendContext: BackendContext,
    fs: FS,
    contexts: IndexedSeq[(Array[Byte], Int)],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None,
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {
    val sparkDeps =
      for {
        rvdDep <- dependency.toIndexedSeq
        dep <- rvdDep.deps
      } yield new AnonymousDependency(dep.asInstanceOf[RVDDependency].rvd.crdd.rdd)

    val rdd =
      new RDD[(Try[Array[Byte]], Int)](sc, sparkDeps) {

        /* Spark insists that `Partition.index` is indeed the index that partition appears in the
         * result of `RDD.getPartitions`.
         *
         * We accept contexts in the form (data, index) and return results in the form (result,
         * index). The index is the index of input context in the original array of contexts. This
         * function may receive a subset of those contexts when retrying queries. We can't use it as
         * the RDD Partition index, therefore; instead store it as a "tag" and use it to transform
         * the RDD result.
         *
         * See `BackendUtils.collectDArray` for how the index is generated. */
        case class TaggedRDDPartition(data: Array[Byte], tag: Int, index: Int) extends Partition

        override protected def getPartitions: Array[Partition] =
          for {
            ((data, index), rddIndex) <- contexts.zipWithIndex.toArray
          } yield TaggedRDDPartition(data, index, rddIndex)

        override def compute(partition: Partition, context: TaskContext)
          : Iterator[(Try[Array[Byte]], Int)] = {
          val sp = partition.asInstanceOf[TaggedRDDPartition]
          val fs = new HadoopFS(null)
          // FIXME: this is broken: the partitionId of SparkTaskContext will be incorrect
          val result =
            try
              Success(f(sp.data, SparkTaskContext.get(), theHailClassLoaderForSparkWorkers, fs))
            catch {
              case NonFatal(exc) =>
                exc.getStackTrace() // Calling getStackTrace appears to ensure the exception is
                // serialized with its stack trace.
                Failure(exc)
            }
          Iterator.single((result, sp.tag))
        }
      }

    val buffer = new ArrayBuffer[(Array[Byte], Int)](contexts.length)
    rdd.collect().foldLeft((Option.empty[Throwable], buffer)) {
      case ((err, buffer), (Success(v), index)) => (err, buffer += ((v, index)))
      case ((err, buffer), (Failure(t), _)) => (err.orElse(Some(t)), buffer)
    }
  }

  def defaultParallelism: Int = sc.defaultParallelism

  override def asSpark(op: String): SparkBackend = this

  def stop(): Unit = SparkBackend.stop()

  def startProgressBar(): Unit =
    ProgressBarBuilder.build(sc)

  private[this] def executionResultToAnnotation(
    ctx: ExecuteContext,
    result: Either[Unit, (PTuple, Long)],
  ) = result match {
    case Left(x) => x
    case Right((pt, off)) => SafeRow(pt, off).get(0)
  }

  def jvmLowerAndExecute(
    ctx: ExecuteContext,
    timer: ExecutionTimer,
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None,
  ): Any = {
    val l = _jvmLowerAndExecute(ctx, ir0, optimize, lowerTable, lowerBM, print)
    executionResultToAnnotation(ctx, l)
  }

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] = {
    val typesToLower: DArrayLowering.Type = (lowerTable, lowerBM) match {
      case (true, true) => DArrayLowering.All
      case (true, false) => DArrayLowering.TableOnly
      case (false, true) => DArrayLowering.BMOnly
      case (false, false) => throw new LowererUnsupportedOperation("no lowering enabled")
    }
    val ir = LoweringPipeline.darrayLowerer(optimize)(typesToLower).apply(ctx, ir0).asInstanceOf[IR]

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ctx, ir)}")

    val res = ir.typ match {
      case TVoid =>
        val (_, f) = ctx.timer.time("Compile") {
          Compile[AsmFunction1RegionUnit](
            ctx,
            FastSeq(),
            FastSeq(classInfo[Region]),
            UnitInfo,
            ir,
            print = print,
          )
        }
        ctx.timer.time("Run")(Left(ctx.scopedExecution((hcl, fs, htc, r) =>
          f(hcl, fs, htc, r).apply(r)
        )))

      case _ =>
        val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) = ctx.timer.time("Compile") {
          Compile[AsmFunction1RegionLong](
            ctx,
            FastSeq(),
            FastSeq(classInfo[Region]),
            LongInfo,
            MakeTuple.ordered(FastSeq(ir)),
            print = print,
          )
        }
        ctx.timer.time("Run")(Right((
          pt,
          ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r).apply(r)),
        )))
    }

    res
  }

  def execute(timer: ExecutionTimer, ir: IR, optimize: Boolean): Any =
    withExecuteContext(timer) { ctx =>
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      val l = _execute(ctx, ir, optimize)
      val javaObjResult =
        ctx.timer.time("convertRegionValueToAnnotation")(executionResultToAnnotation(ctx, l))
      log.info(s"finished execution of query $queryID")
      javaObjResult
    }

  private[this] def _execute(ctx: ExecuteContext, ir: IR, optimize: Boolean)
    : Either[Unit, (PTuple, Long)] = {
    TypeCheck(ctx, ir)
    Validate(ir)
    ctx.irMetadata = ctx.irMetadata.copy(semhash = SemanticHash(ctx)(ir))
    try {
      val lowerTable = getFlag("lower") != null
      val lowerBM = getFlag("lower_bm") != null
      _jvmLowerAndExecute(ctx, ir, optimize, lowerTable, lowerBM)
    } catch {
      case e: LowererUnsupportedOperation if getFlag("lower_only") != null => throw e
      case _: LowererUnsupportedOperation =>
        CompileAndEvaluate._apply(ctx, ir, optimize = optimize)
    }
  }

  def executeLiteral(irStr: String): Int = {
    ExecutionTimer.logTime("SparkBackend.executeLiteral") { timer =>
      withExecuteContext(timer) { ctx =>
        val ir = IRParser.parse_value_ir(irStr, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
        val t = ir.typ
        assert(t.isRealizable)
        val queryID = Backend.nextID()
        log.info(s"starting execution of query $queryID} of initial size ${IRSize(ir)}")
        val retVal = _execute(ctx, ir, true)
        val literalIR = retVal match {
          case Left(_) => throw new HailException("Can't create literal")
          case Right((pt, addr)) =>
            GetFieldByIdx(EncodedLiteral.fromPTypeAndAddress(pt, addr, ctx), 0)
        }
        log.info(s"finished execution of query $queryID")
        addJavaIR(literalIR)
      }
    }
  }

  override def execute(
    ir: String,
    timed: Boolean,
  )(
    consume: (ExecuteContext, Either[Unit, (PTuple, Long)], String) => Unit
  ): Unit = {
    withExecuteContext("SparkBackend.execute") { ctx =>
      val res = ctx.timer.time("execute") {
        val irData =
          IRParser.parse_value_ir(ir, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
        val queryID = Backend.nextID()
        log.info(s"starting execution of query $queryID of initial size ${IRSize(irData)}")
        _execute(ctx, irData, true)
      }
      ctx.timer.finish()
      val timings = if (timed)
        Serialization.write(Map("timings" -> ctx.timer.toMap))(new DefaultFormats {})
      else ""
      consume(ctx, res, timings)
    }
  }

  def encodeToBytes(ctx: ExecuteContext, t: PTuple, off: Long, bufferSpecString: String)
    : Array[Byte] = {
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    assert(t.size == 1)
    val elementType = t.fields(0).typ
    val codec = TypedCodecSpec(
      EType.fromPythonTypeEncoding(elementType.virtualType),
      elementType.virtualType,
      bs,
    )
    assert(t.isFieldDefined(off, 0))
    codec.encode(ctx, elementType, t.loadField(off, 0))
  }

  def pyFromDF(df: DataFrame, jKey: java.util.List[String]): (Int, String) = {
    ExecutionTimer.logTime("SparkBackend.pyFromDF") { timer =>
      val key = jKey.asScala.toArray.toFastSeq
      val signature =
        SparkAnnotationImpex.importType(df.schema).setRequired(true).asInstanceOf[PStruct]
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        val tir = TableLiteral(
          TableValue(
            ctx,
            signature.virtualType.asInstanceOf[TStruct],
            key,
            df.rdd,
            Some(signature),
          ),
          ctx.theHailClassLoader,
        )
        val id = addJavaIR(tir)
        (id, JsonMethods.compact(tir.typ.toJSON))
      }
    }
  }

  def pyToDF(s: String): DataFrame = {
    ExecutionTimer.logTime("SparkBackend.pyToDF") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        val tir = IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
        Interpret(tir, ctx).toDF()
      }
    }
  }

  def pyReadMultipleMatrixTables(jsonQuery: String): java.util.List[MatrixIR] = {
    log.info("pyReadMultipleMatrixTables: got query")
    val kvs = JsonMethods.parse(jsonQuery) match {
      case json4s.JObject(values) => values.toMap
    }

    val paths = kvs("paths").asInstanceOf[json4s.JArray].arr.toArray.map { case json4s.JString(s) =>
      s
    }

    val intervalPointType = parseType(kvs("intervalPointType").asInstanceOf[json4s.JString].s)
    val intervalObjects =
      JSONAnnotationImpex.importAnnotation(kvs("intervals"), TArray(TInterval(intervalPointType)))
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

  def pyAddReference(jsonConfig: String): Unit = addReference(ReferenceGenome.fromJSON(jsonConfig))
  def pyRemoveReference(name: String): Unit = removeReference(name)

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit =
    ExecutionTimer.logTime("SparkBackend.pyReferenceAddLiftover") { timer =>
      withExecuteContext(timer)(ctx => references(name).addLiftover(ctx, chainFile, destRGName))
    }

  def pyRemoveLiftover(name: String, destRGName: String) =
    references(name).removeLiftover(destRGName)

  def pyFromFASTAFile(
    name: String,
    fastaFile: String,
    indexFile: String,
    xContigs: java.util.List[String],
    yContigs: java.util.List[String],
    mtContigs: java.util.List[String],
    parInput: java.util.List[String],
  ): String = {
    ExecutionTimer.logTime("SparkBackend.pyFromFASTAFile") { timer =>
      withExecuteContext(timer) { ctx =>
        val rg = ReferenceGenome.fromFASTAFile(
          ctx,
          name,
          fastaFile,
          indexFile,
          xContigs.asScala.toArray,
          yContigs.asScala.toArray,
          mtContigs.asScala.toArray,
          parInput.asScala.toArray,
        )
        rg.toJSONString
      }
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit =
    ExecutionTimer.logTime("SparkBackend.pyAddSequence") { timer =>
      withExecuteContext(timer)(ctx => references(name).addSequence(ctx, fastaFile, indexFile))
    }

  def pyRemoveSequence(name: String) = references(name).removeSequence()

  def pyExportBlockMatrix(
    pathIn: String,
    pathOut: String,
    delimiter: String,
    header: String,
    addIndex: Boolean,
    exportType: String,
    partitionSize: java.lang.Integer,
    entries: String,
  ): Unit = {
    ExecutionTimer.logTime("SparkBackend.pyExportBlockMatrix") { timer =>
      withExecuteContext(timer) { ctx =>
        val rm = RowMatrix.readBlockMatrix(fs, pathIn, partitionSize)
        entries match {
          case "full" =>
            rm.export(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
          case "lower" =>
            rm.exportLowerTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
          case "strict_lower" =>
            rm.exportStrictLowerTriangle(
              ctx,
              pathOut,
              delimiter,
              Option(header),
              addIndex,
              exportType,
            )
          case "upper" =>
            rm.exportUpperTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
          case "strict_upper" =>
            rm.exportStrictUpperTriangle(
              ctx,
              pathOut,
              delimiter,
              Option(header),
              addIndex,
              exportType,
            )
        }
      }
    }
  }

  def pyFitLinearMixedModel(lmm: LinearMixedModel, pa_t: RowMatrix, a_t: RowMatrix): TableIR =
    ExecutionTimer.logTime("SparkBackend.pyAddSequence") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        lmm.fit(ctx, pa_t, Option(a_t))
      }
    }

  def parse_value_ir(s: String, refMap: java.util.Map[String, String]): IR =
    ExecutionTimer.logTime("SparkBackend.parse_value_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_value_ir(
          s,
          IRParserEnvironment(ctx, irMap = persistedIR.toMap),
          BindingEnv.eval(refMap.asScala.toMap.mapValues(IRParser.parseType).toSeq: _*),
        )
      }
    }

  def parse_table_ir(s: String): TableIR =
    ExecutionTimer.logTime("SparkBackend.parse_table_ir") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      }
    }

  def parse_matrix_ir(s: String): MatrixIR =
    ExecutionTimer.logTime("SparkBackend.parse_matrix_ir") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      }
    }

  def parse_blockmatrix_ir(s: String): BlockMatrixIR =
    ExecutionTimer.logTime("SparkBackend.parse_blockmatrix_ir") { timer =>
      withExecuteContext(timer, selfContainedExecution = false) { ctx =>
        IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      }
    }

  override def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader = {
    if (getFlag("use_new_shuffle") != null)
      return LowerDistributedSort.distributedSort(ctx, stage, sortFields, rt)

    val (globals, rvd) = TableStageToRVD(ctx, stage)
    val globalsLit = globals.toEncodedLiteral(ctx.theHailClassLoader)

    if (sortFields.forall(_.sortOrder == Ascending)) {
      return RVDTableReader(rvd.changeKey(ctx, sortFields.map(_.field)), globalsLit, rt)
    }

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
    val rdd = rvd.keyedEncodedRDD(ctx, codec, sortFields.map(_.field)).sortBy(
      _._1,
      numPartitions = nPartitions.getOrElse(rvd.getNumPartitions),
    )(ord, act)
    val (rowPType: PStruct, orderedCRDD) = codec.decodeRDD(ctx, rowType, rdd.map(_._2))
    RVDTableReader(RVD.unkeyed(rowPType, orderedCRDD), globalsLit, rt)
  }

  def close(): Unit =
    longLifeTempFileManager.cleanup()

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage = {
    CanLowerEfficiently(ctx, inputIR) match {
      case Some(failReason) =>
        log.info(s"SparkBackend: could not lower IR to table stage: $failReason")
        inputIR.analyzeAndExecute(ctx).asTableStage(ctx)
      case None =>
        LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
    }
  }
}

case class SparkBackendComputeRDDPartition(data: Array[Byte], index: Int) extends Partition

class SparkBackendComputeRDD(
  sc: SparkContext,
  @transient private val collection: Array[Array[Byte]],
  f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte],
  deps: Seq[Dependency[_]],
) extends RDD[Array[Byte]](sc, deps) {

  override def getPartitions: Array[Partition] =
    Array.tabulate(collection.length)(i => SparkBackendComputeRDDPartition(collection(i), i))

  override def compute(partition: Partition, context: TaskContext): Iterator[Array[Byte]] = {
    val sp = partition.asInstanceOf[SparkBackendComputeRDDPartition]
    val fs = new HadoopFS(null)
    Iterator.single(f(sp.data, SparkTaskContext.get(), theHailClassLoaderForSparkWorkers, fs))
  }
}
