package is.hail.backend.spark

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend._
import is.hail.backend.py4j.Py4JBackendExtensions
import is.hail.expr.Validate
import is.hail.expr.ir._
import is.hail.expr.ir.LoweredTableReader.LoweredTableReaderCoercer
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs.MakeTuple
import is.hail.expr.ir.lowering._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs._
import is.hail.linalg.BlockMatrix
import is.hail.rvd.RVD
import is.hail.types._
import is.hail.types.physical.{PStruct, PTuple}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.ExecutionException
import scala.reflect.ClassTag
import scala.util.control.NonFatal

import java.io.PrintWriter

import com.fasterxml.jackson.core.StreamReadConstraints
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import sourcecode.Enclosing

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
  object Flags {
    val MaxStageParallelism = "spark_max_stage_parallelism"
  }

  // From https://github.com/hail-is/hail/issues/14580 :
  //   IR can get quite big, especially as it can contain an arbitrary
  //   amount of encoded literals from the user's python session. This
  //   was a (controversial) restriction imposed by Jackson and should be lifted.
  //
  // We remove this restriction at the earliest point possible for each backend/
  // This can't be unified since each backend has its own entry-point from python
  // and its own specific initialisation code.
  StreamReadConstraints.overrideDefaultStreamReadConstraints(
    StreamReadConstraints.builder().maxStringLength(Integer.MAX_VALUE).build()
  )

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
      new SparkBackend(
        tmpdir,
        localTmpdir,
        sc1,
        mutable.Map(ReferenceGenome.builtinReferences().toSeq: _*),
        gcsRequesterPaysProject,
        gcsRequesterPaysBuckets,
      )
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
  override val references: mutable.Map[String, ReferenceGenome],
  gcsRequesterPaysProject: String,
  gcsRequesterPaysBuckets: String,
) extends Backend with Py4JBackendExtensions {

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

  override def backend: Backend = this
  override val flags: HailFeatureFlags = HailFeatureFlags.fromEnv()

  override val longLifeTempFileManager: TempFileManager =
    new OwningTempFileManager(fs)

  private[this] val bmCache = mutable.Map.empty[String, BlockMatrix]
  private[this] val codeCache = new Cache[CodeCacheKey, CompiledFunction[_]](50)
  private[this] val persistedIr = mutable.Map.empty[Int, BaseIR]
  private[this] val coercerCache = new Cache[Any, LoweredTableReaderCoercer](32)

  def createExecuteContextForTests(
    timer: ExecutionTimer,
    region: Region,
    selfContainedExecution: Boolean = true,
  ): ExecuteContext =
    new ExecuteContext(
      tmpdir,
      localTmpdir,
      this,
      references.toMap,
      fs,
      region,
      timer,
      if (selfContainedExecution) null else NonOwningTempFileManager(longLifeTempFileManager),
      theHailClassLoader,
      flags,
      new BackendContext {
        override val executionCache: ExecutionCache =
          ExecutionCache.forTesting
      },
      new IrMetadata(),
      ImmutableMap.empty,
      ImmutableMap.empty,
      ImmutableMap.empty,
      ImmutableMap.empty,
    )

  override def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): T =
    ExecutionTimer.logTime { timer =>
      ExecuteContext.scoped(
        tmpdir,
        localTmpdir,
        this,
        references.toMap,
        fs,
        timer,
        null,
        theHailClassLoader,
        flags,
        new BackendContext {
          override val executionCache: ExecutionCache =
            ExecutionCache.fromFlags(flags, fs, tmpdir)
        },
        new IrMetadata(),
        bmCache,
        codeCache,
        persistedIr,
        coercerCache,
      )(f)
    }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] =
    new SparkBroadcastValue[T](sc.broadcast(value))

  override def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    contexts: IndexedSeq[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency],
    partitions: Option[IndexedSeq[Int]],
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {
    val sparkDeps =
      for {
        rvdDep <- dependency.toIndexedSeq
        dep <- rvdDep.deps
      } yield new AnonymousDependency(dep.asInstanceOf[RVDDependency].rvd.crdd.rdd)

    val rdd =
      new RDD[Array[Byte]](sc, sparkDeps) {

        case class RDDPartition(data: Array[Byte], override val index: Int) extends Partition

        override protected val getPartitions: Array[Partition] =
          Array.tabulate(contexts.length)(index => RDDPartition(contexts(index), index))

        override def compute(partition: Partition, context: TaskContext): Iterator[Array[Byte]] = {
          val sp = partition.asInstanceOf[RDDPartition]
          val fs = new HadoopFS(null)
          Iterator.single(f(sp.data, SparkTaskContext.get(), theHailClassLoaderForSparkWorkers, fs))
        }
      }

    val chunkSize = flags.get(SparkBackend.Flags.MaxStageParallelism).toInt
    val partsToRun = partitions.getOrElse(contexts.indices)
    val buffer = new ArrayBuffer[(Array[Byte], Int)](partsToRun.length)
    var failure: Option[Throwable] = None

    try {
      for (subparts <- partsToRun.grouped(chunkSize)) {
        sc.runJob(
          rdd,
          (_: TaskContext, it: Iterator[Array[Byte]]) => it.next(),
          subparts,
          (idx, result: Array[Byte]) => buffer += result -> subparts(idx),
        )
      }
    } catch {
      case e: ExecutionException => failure = failure.orElse(Some(e.getCause))
      case NonFatal(t) => failure = failure.orElse(Some(t))
    }

    (failure, buffer.sortBy(_._2))
  }

  def defaultParallelism: Int = sc.defaultParallelism

  override def asSpark(op: String): SparkBackend = this

  def close(): Unit = {
    longLifeTempFileManager.close()
    SparkBackend.stop()
  }

  def startProgressBar(): Unit =
    ProgressBarBuilder.build(sc)

  def jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None,
  ): Any =
    _jvmLowerAndExecute(ctx, ir0, optimize, lowerTable, lowerBM, print) match {
      case Left(x) => x
      case Right((pt, off)) => SafeRow(pt, off).get(0)
    }

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] =
    ctx.time {
      val typesToLower: DArrayLowering.Type = (lowerTable, lowerBM) match {
        case (true, true) => DArrayLowering.All
        case (true, false) => DArrayLowering.TableOnly
        case (false, true) => DArrayLowering.BMOnly
        case (false, false) => throw new LowererUnsupportedOperation("no lowering enabled")
      }
      val ir =
        LoweringPipeline.darrayLowerer(optimize)(typesToLower).apply(ctx, ir0).asInstanceOf[IR]

      if (!Compilable(ir))
        throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ctx, ir)}")

      ir.typ match {
        case TVoid =>
          val (_, f) = Compile[AsmFunction1RegionUnit](
            ctx,
            FastSeq(),
            FastSeq(classInfo[Region]),
            UnitInfo,
            ir,
            print = print,
          )

          Left(ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r)(r)))
        case _ =>
          val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) =
            Compile[AsmFunction1RegionLong](
              ctx,
              FastSeq(),
              FastSeq(classInfo[Region]),
              LongInfo,
              MakeTuple.ordered(FastSeq(ir)),
              print = print,
            )

          Right((pt, ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r)(r))))
      }
    }

  override def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      TypeCheck(ctx, ir)
      Validate(ir)
      ctx.irMetadata.semhash = SemanticHash(ctx)(ir)
      try {
        val lowerTable = flags.get("lower") != null
        val lowerBM = flags.get("lower_bm") != null
        _jvmLowerAndExecute(ctx, ir, optimize = true, lowerTable, lowerBM)
      } catch {
        case e: LowererUnsupportedOperation if flags.get("lower_only") != null => throw e
        case _: LowererUnsupportedOperation =>
          CompileAndEvaluate._apply(ctx, ir, optimize = true)
      }
    }

  override def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader = {
    if (flags.get("use_new_shuffle") != null)
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

    val codec = TypedCodecSpec(ctx, rvd.rowPType, BufferSpec.wireSpec)
    val rdd = rvd.keyedEncodedRDD(ctx, codec, sortFields.map(_.field)).sortBy(
      _._1,
      numPartitions = nPartitions.getOrElse(rvd.getNumPartitions),
    )(ord, act)
    val (rowPType: PStruct, orderedCRDD) = codec.decodeRDD(ctx, rowType, rdd.map(_._2))
    RVDTableReader(RVD.unkeyed(rowPType, orderedCRDD), globalsLit, rt)
  }

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage = {
    CanLowerEfficiently(ctx, inputIR) match {
      case Some(failReason) =>
        log.info(s"SparkBackend: could not lower IR to table stage: $failReason")
        ExecuteRelational(ctx, inputIR).asTableStage(ctx)
      case None =>
        LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
    }
  }
}
