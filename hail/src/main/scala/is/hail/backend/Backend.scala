package is.hail.backend

import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{ExecuteContext, SortField}
import is.hail.expr.ir.lowering.TableStage
import is.hail.types.virtual.Type
import is.hail.utils._

import scala.reflect.ClassTag

abstract class BroadcastValue[T] { def value: T }

abstract class BackendContext

abstract class Backend {
  def defaultParallelism: Int

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex(backendContext: BackendContext, collection: Array[Array[Byte]])(f: (Array[Byte], Int) => Array[Byte]): Array[Array[Byte]]

  def stop(): Unit

  def asSpark(op: String): SparkBackend =
    fatal(s"${ getClass.getSimpleName }: $op requires SparkBackend")

  def lowerDistributedSort(ctx: ExecuteContext, stage: TableStage, sortFields: IndexedSeq[SortField], relationalLetsAbove: Seq[(String, Type)]): TableStage
}

