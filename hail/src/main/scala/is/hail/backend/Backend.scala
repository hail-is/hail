package is.hail.backend

import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.SortField
import is.hail.expr.ir.lowering.TableStage
import is.hail.utils._

import scala.reflect.ClassTag

abstract class BroadcastValue[T] { def value: T }

abstract class Backend {
  def defaultParallelism: Int

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U]

  def stop(): Unit

  def asSpark(op: String): SparkBackend =
    fatal(s"${ getClass.getSimpleName }: $op requires SparkBackend")

  def lowerDistributedSort(stage: TableStage, sortFields: IndexedSeq[SortField]): TableStage = throw new UnsupportedOperationException
}

