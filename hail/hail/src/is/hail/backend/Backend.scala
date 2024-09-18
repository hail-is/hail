package is.hail.backend

import is.hail.asm4s.HailClassLoader
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{IR, LoweringAnalyses, SortField, TableIR, TableReader}
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs.FS
import is.hail.types.RTable
import is.hail.types.encoded.EType
import is.hail.types.physical.PTuple
import is.hail.utils.ExecutionTimer.Timings
import is.hail.utils.fatal

import scala.reflect.ClassTag

import java.io.{Closeable, OutputStream}

import sourcecode.Enclosing

object Backend {

  private var id: Long = 0L

  def nextID(): String = {
    id += 1
    s"hail_query_$id"
  }

  def encodeToOutputStream(
    ctx: ExecuteContext,
    t: PTuple,
    off: Long,
    bufferSpec: BufferSpec,
    os: OutputStream,
  ): Unit = {
    assert(t.size == 1)
    val elementType = t.fields(0).typ
    val codec = TypedCodecSpec(
      EType.fromPythonTypeEncoding(elementType.virtualType),
      elementType.virtualType,
      bufferSpec,
    )
    assert(t.isFieldDefined(off, 0))
    codec.encode(ctx, elementType, t.loadField(off, 0), os)
  }
}

abstract class BroadcastValue[T] { def value: T }

trait BackendContext {
  def executionCache: ExecutionCache
}

abstract class Backend extends Closeable {

  def defaultParallelism: Int

  def canExecuteParallelTasksOnDriver: Boolean = true

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    contexts: IndexedSeq[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None,
    partitions: Option[IndexedSeq[Int]] = None,
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)])

  def asSpark(implicit E: Enclosing): SparkBackend =
    fatal(s"${getClass.getSimpleName}: ${E.value} requires SparkBackend")

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
  ): TableReader =
    lowerDistributedSort(ctx, stage, sortFields, rt, None)

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    inputIR: TableIR,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int] = None,
  ): TableReader = {
    val analyses = LoweringAnalyses.apply(inputIR, ctx)
    val inputStage = tableToTableStage(ctx, inputIR, analyses)
    lowerDistributedSort(ctx, inputStage, sortFields, rt, nPartitions)
  }

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage

  def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): (T, Timings)

  def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)]
}
