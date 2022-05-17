package is.hail.backend

import is.hail.asm4s._
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.{IR, SortField}
import is.hail.io.fs.FS
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.utils._

import scala.reflect.ClassTag

object Backend {

  private var id: Long = 0L
  def nextID(): String = {
    id += 1
    s"hail_query_$id"
  }
}

abstract class BroadcastValue[T] { def value: T }

abstract class BackendContext

abstract class Backend {
  def defaultParallelism: Int

  def canExecuteParallelTasksOnDriver: Boolean = true

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit

  def unpersist(backendContext: BackendContext, id: String): Unit

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType

  def parallelizeAndComputeWithIndex(backendContext: BackendContext, fs: FS, collection: Array[Array[Byte]], dependency: Option[TableStageDependency] = None)(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]): Array[Array[Byte]]

  def stop(): Unit

  def asSpark(op: String): SparkBackend =
    fatal(s"${ getClass.getSimpleName }: $op requires SparkBackend")

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): TableStage
}
