package is.hail.backend.local

import is.hail.CancellingExecutorService
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.Validate
import is.hail.expr.ir._
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.io.fs._
import is.hail.types._
import is.hail.types.physical.PTuple
import is.hail.utils._

import scala.reflect.ClassTag

import java.io.PrintWriter

import com.fasterxml.jackson.core.StreamReadConstraints
import com.google.common.util.concurrent.MoreExecutors

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

class LocalTaskContext(val partitionId: Int, val stageId: Int) extends HailTaskContext {
  override def attemptNumber(): Int = 0
}

object LocalBackend extends Backend {

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
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    skipLoggingConfiguration: Boolean = false,
  ): LocalBackend.type =
    synchronized {
      if (!skipLoggingConfiguration) Logging.configureLogging(logFile, quiet, append)
      this
    }

  private case class Context(hcl: HailClassLoader, override val executionCache: ExecutionCache)
      extends BackendContext

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = new LocalBroadcastValue[T](value)

  private[this] var stageIdx: Int = 0

  private[this] def nextStageId(): Int =
    synchronized {
      val current = stageIdx
      stageIdx += 1
      current
    }

  override def parallelizeAndComputeWithIndex(
    ctx: BackendContext,
    fs: FS,
    contexts: IndexedSeq[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency],
    partitions: Option[IndexedSeq[Int]],
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {

    val stageId = nextStageId()
    val hcl = ctx.asInstanceOf[Context].hcl
    runAllKeepFirstError(new CancellingExecutorService(MoreExecutors.newDirectExecutorService())) {
      partitions.getOrElse(contexts.indices).map { i =>
        (
          () => using(new LocalTaskContext(i, stageId))(f(contexts(i), _, hcl, fs)),
          i,
        )
      }
    }
  }

  override def backendContext(ctx: ExecuteContext): BackendContext =
    Context(ctx.theHailClassLoader, ExecutionCache.fromFlags(ctx.flags, ctx.fs, ctx.tmpdir))

  def defaultParallelism: Int = 1

  def close(): Unit =
    synchronized { stageIdx = 0 }

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] =
    CompileAndEvaluate._apply(
      ctx,
      ir0,
      LoweringPipeline.darrayLowerer(DArrayLowering.All),
      print = print,
    )

  override def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      TypeCheck(ctx, ir)
      Validate(ir)
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      ctx.irMetadata.semhash = SemanticHash(ctx, ir)
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
