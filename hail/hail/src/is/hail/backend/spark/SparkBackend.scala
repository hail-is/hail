package is.hail.backend.spark

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend._
import is.hail.backend.Backend.PartitionFn
import is.hail.expr.Validate
import is.hail.expr.ir._
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs._
import is.hail.rvd.RVD
import is.hail.types._
import is.hail.types.physical.{PStruct, PTuple}
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.mutable
import scala.concurrent.{CancellationException, ExecutionException}
import scala.reflect.ClassTag
import scala.util.control.NonFatal

import java.io.PrintWriter

import com.fasterxml.jackson.core.StreamReadConstraints
import org.apache.hadoop
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
        sparkTC.addTaskCompletionListener[Unit]((_: TaskContext) => SparkTaskContext.finish()): Unit

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

object SparkBackend extends Logging {
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

  is.hail.linalg.registerImplOpMulMatrix_DMD_DVD_eq_DVD

  private var theSparkBackend: SparkBackend = _

  def sparkContext(implicit E: Enclosing): SparkContext =
    synchronized {
      if (theSparkBackend == null) throw new IllegalStateException(E.value)
      else theSparkBackend.sc
    }

  def checkSparkCompatibility(jarVersion: String, sparkVersion: String): Unit = {
    def majorMinor(version: String): String = version.split("\\.", 3).take(2).mkString(".")

    if (majorMinor(jarVersion) != majorMinor(sparkVersion))
      fatal(
        s"This Hail JAR was compiled for Spark $jarVersion, cannot run with Spark $sparkVersion.\n" +
          s"  The major and minor versions must agree, though the patch version can differ."
      )
    else if (jarVersion != sparkVersion)
      logger.warn(
        s"This Hail JAR was compiled for Spark $jarVersion, running with Spark $sparkVersion.\n" +
          s"  Compatibility is not guaranteed."
      )
  }

  def createSparkConf(appName: String, master: String, local: String, blockSize: Long)
    : SparkConf = {
    require(blockSize >= 0)

    checkSparkCompatibility(is.hail.SparkVersion, org.apache.spark.SPARK_VERSION)

    val conf = new SparkConf().setAppName(appName)

    if (master != null) conf.setMaster(master): Unit
    else if (!conf.contains("spark.master")) conf.setMaster(local): Unit

    conf
      .set("spark.logConf", "true")
      .set("spark.kryoserializer.buffer.max", "1g")
      .set("spark.driver.maxResultSize", "0")
      .set(
        "spark.hadoop.io.compression.codecs",
        "org.apache.hadoop.io.compress.DefaultCodec," +
          "is.hail.io.compress.BGzipCodec," +
          "is.hail.io.compress.BGzipCodecTbi," +
          "org.apache.hadoop.io.compress.GzipCodec",
      )
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "is.hail.kryo.HailKryoRegistrator")
      .set(
        "spark.hadoop.mapreduce.input.fileinputformat.split.minsize",
        (blockSize * 1024L * 1024L).toString,
      ): Unit

    // load additional Spark properties from HAIL_SPARK_PROPERTIES
    sys.env.get("HAIL_SPARK_PROPERTIES").foreach { hailSparkProperties =>
      for (p <- hailSparkProperties.split(",")) {
        p.split("=") match {
          case Array(k, v) =>
            logger.info(s"set Spark property from HAIL_SPARK_PROPERTIES: $k=$v")
            conf.set(k, v)
          case _ =>
            logger.warn(s"invalid key-value property pair in HAIL_SPARK_PROPERTIES: $p")
        }
      }
    }

    conf
  }

  def pySparkSession(
    appName: String = "Hail",
    master: String = null,
    local: String = "local[*]",
    minBlockSize: Long = 1L,
  ): SparkSession = {
    val conf = createSparkConf(appName, master, local, minBlockSize)
    SparkSession.builder().config(conf).getOrCreate()
  }

  def checkSparkConfiguration(conf: SparkConf): Unit = {
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

  private[this] lazy val CompressionCodecs: Array[String] =
    Array(
      "org.apache.hadoop.io.compress.DefaultCodec",
      "is.hail.io.compress.BGzipCodec",
      "is.hail.io.compress.BGzipCodecTbi",
      "org.apache.hadoop.io.compress.GzipCodec",
    )

  def getOrCreate(session: SparkSession): SparkBackend =
    synchronized {
      if (theSparkBackend == null) SparkBackend(session)
      else {
        // there should be only one SparkContext
        if (session.sparkContext ne theSparkBackend.sc)
          fatal(
            "Spark requires that there is at most one active `SparkContext` per JVM.\n" +
              "You must stop() the active context or specify it as the `sc` parameter to hl.init().\n" +
              "Please refer to the pyspark documentation for how to obtain the active spark context."
          )
        theSparkBackend
      }
    }

  def apply(session: SparkSession): SparkBackend =
    synchronized {
      require(theSparkBackend == null)
      val sc = session.sparkContext
      sc.hadoopConfiguration.set("io.compression.codecs", CompressionCodecs.mkString(","))
      checkSparkConfiguration(sc.getConf)
      sc.uiWebUrl.foreach(ui => logger.info(s"SparkUI: $ui"))
      theSparkBackend = new SparkBackend(session)
      theSparkBackend
    }
}

// This indicates a narrow (non-shuffle) dependency on _rdd. It works since narrow dependency `getParents`
// is only used to compute preferred locations, which is something we don't need to worry about
class AnonymousDependency[T](val _rdd: RDD[T]) extends NarrowDependency[T](_rdd) {
  override def getParents(partitionId: Int): Seq[Int] = Seq.empty
}

class SparkBackend(val spark: SparkSession) extends Backend with Logging {

  // cached for convenience
  val sc: SparkContext =
    spark.sparkContext

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] =
    new SparkBroadcastValue[T](sc.broadcast(value))

  override def runtimeContext(ctx: ExecuteContext): DriverRuntimeContext =
    new DriverRuntimeContext {

      override val executionCache: ExecutionCache =
        ExecutionCache.fromFlags(ctx.flags, ctx.fs, ctx.tmpdir)

      override def mapCollectPartitions(
        globals: Array[Byte],
        contexts: IndexedSeq[Array[Byte]],
        stageIdentifier: String,
        dependency: Option[TableStageDependency],
        partitions: Option[IndexedSeq[Int]],
      )(
        f: PartitionFn
      ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {
        val sparkDeps =
          for {
            rvdDep <- dependency.toIndexedSeq
            dep <- rvdDep.deps
          } yield new AnonymousDependency(dep.asInstanceOf[RVDDependency].rvd.crdd.rdd)

        val rdd: RDD[Array[Byte]] =
          new RDD[Array[Byte]](sc, sparkDeps) {

            case class RDDPartition(data: Array[Byte], override val index: Int) extends Partition

            val fsConfig: SerializableHadoopConfiguration =
              ctx.fs.getConfiguration().asInstanceOf[SerializableHadoopConfiguration]

            override protected val getPartitions: Array[Partition] =
              Array.tabulate(contexts.length)(index => RDDPartition(contexts(index), index))

            override def compute(partition: Partition, context: TaskContext)
              : Iterator[Array[Byte]] =
              Iterator.single(
                f(
                  globals,
                  partition.asInstanceOf[RDDPartition].data,
                  SparkTaskContext.get(),
                  theHailClassLoaderForSparkWorkers,
                  new HadoopFS(fsConfig),
                )
              )
          }

        val todo: IndexedSeq[Int] =
          partitions.getOrElse(contexts.indices)

        val buffer = ArraySeq.newBuilder[(Array[Byte], Int)]
        buffer.sizeHint(todo.length)

        var failure: Option[Throwable] =
          None

        val maxStageParallelism =
          ctx.flags.get(SparkBackend.Flags.MaxStageParallelism).toInt

        sc.setJobGroup(stageIdentifier, "", interruptOnCancel = true)
        try {
          for (subparts <- todo.grouped(maxStageParallelism)) {
            sc.runJob(
              rdd,
              (_: TaskContext, it: Iterator[Array[Byte]]) => it.next(),
              subparts,
              (idx, result: Array[Byte]) =>
                // appending here is safe as resultHandler is called in a synchronized block
                buffer += result -> subparts(idx),
            )
          }
        } catch {
          case NonFatal(t) => failure = failure.orElse(Some(t))
          case e: ExecutionException => failure = failure.orElse(Some(e.getCause))
          case _: InterruptedException =>
            sc.cancelJobGroup(stageIdentifier)
            Thread.currentThread().interrupt()
            throw new CancellationException()
        } finally sc.clearJobGroup()

        (failure, buffer.result().sortBy(_._2))
      }
    }

  def defaultParallelism: Int =
    sc.defaultParallelism

  override def asSpark(implicit E: Enclosing): SparkBackend = this

  def close(): Unit =
    SparkBackend.synchronized {
      assert(this eq SparkBackend.theSparkBackend)
      SparkBackend.theSparkBackend = null
      // Hadoop does not honor the hadoop configuration as a component of the cache key for file
      // systems, so we blow away the cache so that a new configuration can successfully take
      // effect.
      // https://github.com/hail-is/hail/pull/12133#issuecomment-1241322443
      hadoop.fs.FileSystem.closeAll()
    }

  def jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None,
  ): Any =
    _jvmLowerAndExecute(ctx, ir0, lowerTable, lowerBM, print) match {
      case Left(x) => x
      case Right((pt, off)) => SafeRow(pt, off).get(0)
    }

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    lowerTable: Boolean,
    lowerBM: Boolean,
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] = {
    val typesToLower: DArrayLowering.Type =
      (lowerTable, lowerBM) match {
        case (true, true) => DArrayLowering.All
        case (true, false) => DArrayLowering.TableOnly
        case (false, true) => DArrayLowering.BMOnly
        case (false, false) => throw new LowererUnsupportedOperation("no lowering enabled")
      }

    CompileAndEvaluate._apply(
      ctx,
      ir0,
      lower = LoweringPipeline.darrayLowerer(typesToLower),
      print = print,
    )
  }

  override def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      TypeCheck(ctx, ir)
      Validate(ir)

      if (ctx.flags.isDefined(ExecutionCache.Flags.UseFastRestarts))
        ctx.irMetadata.semhash = SemanticHash(ctx, ir)

      try {
        val lowerTable = ctx.flags.get("lower") != null
        val lowerBM = ctx.flags.get("lower_bm") != null
        _jvmLowerAndExecute(ctx, ir, lowerTable, lowerBM)
      } catch {
        case e: LowererUnsupportedOperation if ctx.flags.get("lower_only") != null => throw e
        case _: LowererUnsupportedOperation =>
          CompileAndEvaluate._apply(ctx, ir, lower = LoweringPipeline.relationalLowerer)
      }
    }

  override def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader = {
    if (ctx.flags.get("use_new_shuffle") != null)
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
    : TableStage =
    CanLowerEfficiently(ctx, inputIR) match {
      case Some(failReason) =>
        logger.info(s"SparkBackend: could not lower IR to table stage: $failReason")
        ExecuteRelational(ctx, inputIR).asTableStage(ctx)
      case None =>
        LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
    }
}
