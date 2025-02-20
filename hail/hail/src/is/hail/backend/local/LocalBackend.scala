package is.hail.backend.local

import is.hail.{CancellingExecutorService, HailContext, HailFeatureFlags}
import is.hail.annotations.Region
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
import is.hail.io.fs._
import is.hail.types._
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual.TVoid
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.reflect.ClassTag

import java.io.PrintWriter

import com.fasterxml.jackson.core.StreamReadConstraints
import com.google.common.util.concurrent.MoreExecutors
import org.apache.hadoop
import sourcecode.Enclosing

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

class LocalTaskContext(val partitionId: Int, val stageId: Int) extends HailTaskContext {
  override def attemptNumber(): Int = 0
}

object LocalBackend {
  private var theLocalBackend: LocalBackend = _

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

  def apply(
    tmpdir: String,
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    skipLoggingConfiguration: Boolean = false,
  ): LocalBackend = synchronized {
    require(theLocalBackend == null)

    if (!skipLoggingConfiguration)
      HailContext.configureLogging(logFile, quiet, append)

    theLocalBackend = new LocalBackend(
      tmpdir,
      mutable.Map(ReferenceGenome.builtinReferences().toSeq: _*),
    )

    theLocalBackend
  }

  def stop(): Unit = synchronized {
    if (theLocalBackend != null) {
      theLocalBackend = null
      // Hadoop does not honor the hadoop configuration as a component of the cache key for file
      // systems, so we blow away the cache so that a new configuration can successfully take
      // effect.
      // https://github.com/hail-is/hail/pull/12133#issuecomment-1241322443
      hadoop.fs.FileSystem.closeAll()
    }
  }
}

class LocalBackend(
  val tmpdir: String,
  override val references: mutable.Map[String, ReferenceGenome],
) extends Backend with Py4JBackendExtensions {

  override def backend: Backend = this
  override val flags: HailFeatureFlags = HailFeatureFlags.fromEnv()
  override def longLifeTempFileManager: TempFileManager = null

  private[this] val theHailClassLoader = new HailClassLoader(getClass.getClassLoader)
  private[this] val codeCache = new Cache[CodeCacheKey, CompiledFunction[_]](50)
  private[this] val persistedIR: mutable.Map[Int, BaseIR] = mutable.Map()
  private[this] val coercerCache = new Cache[Any, LoweredTableReaderCoercer](32)

  // flags can be set after construction from python
  def fs: FS = RouterFS.buildRoutes(CloudStorageFSConfig.fromFlagsAndEnv(None, flags))

  override def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): T =
    ExecutionTimer.logTime { timer =>
      val fs = this.fs
      ExecuteContext.scoped(
        tmpdir,
        tmpdir,
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
        ImmutableMap.empty,
        codeCache,
        persistedIR,
        coercerCache,
      )(f)
    }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = new LocalBroadcastValue[T](value)

  private[this] var stageIdx: Int = 0

  private[this] def nextStageId(): Int = {
    val current = stageIdx
    stageIdx += 1
    current
  }

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

    val stageId = nextStageId()
    runAllKeepFirstError(new CancellingExecutorService(MoreExecutors.newDirectExecutorService())) {
      partitions.getOrElse(contexts.indices).map { i =>
        (
          () =>
            using(new LocalTaskContext(i, stageId)) {
              f(contexts(i), _, theHailClassLoader, fs)
            },
          i,
        )
      }
    }
  }

  def defaultParallelism: Int = 1

  def close(): Unit = LocalBackend.stop()

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] =
    ctx.time {
      val ir =
        LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, ir0).asInstanceOf[IR]

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
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      ctx.irMetadata.semhash = SemanticHash(ctx)(ir)
      val res = _jvmLowerAndExecute(ctx, ir)
      log.info(s"finished execution of query $queryID")
      res
    }

  override def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader =
    LowerDistributedSort.distributedSort(ctx, stage, sortFields, rt, nPartitions)

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage =
    LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
}
