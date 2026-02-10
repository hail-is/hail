package is.hail.backend.local

import is.hail.backend._
import is.hail.backend.Backend.PartitionFn
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.Validate
import is.hail.expr.ir._
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.types._
import is.hail.types.physical.PTuple
import is.hail.utils._

import scala.reflect.ClassTag
import scala.util.control.NonFatal

import java.io.PrintWriter

import com.fasterxml.jackson.core.StreamReadConstraints

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

object LocalBackend extends Backend with Logging {

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

  override def broadcast[T: ClassTag](value: T): BroadcastValue[T] =
    new LocalBroadcastValue[T](value)

  override def runtimeContext(ctx: ExecuteContext): DriverRuntimeContext =
    new DriverRuntimeContext {

      override val executionCache: ExecutionCache =
        ExecutionCache.fromFlags(ctx.flags, ctx.fs, ctx.localTmpdir)

      override def mapCollectPartitions(
        globals: Array[Byte],
        contexts: IndexedSeq[Array[Byte]],
        stageIdentifier: String,
        dependency: Option[TableStageDependency],
        partitions: Option[IndexedSeq[Int]],
      )(
        f: PartitionFn
      ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {

        val todo: IndexedSeq[Int] =
          partitions.getOrElse(contexts.indices)

        val results = ArraySeq.newBuilder[(Array[Byte], Int)]
        results.sizeHint(todo.length)

        var failure: Option[Throwable] =
          None

        try
          for (idx <- todo)
            results += ctx.scopedExecution { (hcl, fs, ctx, r) =>
              (f(hcl, fs, ctx, r)(globals, contexts(idx)), idx)
            }
        catch {
          case NonFatal(t) =>
            failure = Some(t)
        }

        (failure, results.result())
      }
    }

  override def defaultParallelism: Int = 1

  override def close(): Unit = {}

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
      logger.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      if (ctx.flags.isDefined(ExecutionCache.Flags.UseFastRestarts))
        ctx.irMetadata.semhash = SemanticHash(ctx, ir)

      val res = _jvmLowerAndExecute(ctx, ir)
      logger.info(s"finished execution of query $queryID")
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

  override def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage =
    LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
}
