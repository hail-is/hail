package is.hail.backend.spark

import is.hail.linalg.BlockMatrix
import is.hail.types.BlockMatrixType
import is.hail.utils._

import scala.collection.mutable

case class SparkBlockMatrixCache() {
  private[this] val blockmatrices: mutable.Map[String, BlockMatrix] = new mutable.HashMap()

  def persistBlockMatrix(id: String, value: BlockMatrix, storageLevel: String): Unit =
    blockmatrices.update(id, value.persist(storageLevel))

  def getPersistedBlockMatrix(id: String): BlockMatrix =
    blockmatrices.getOrElse(id, fatal(s"Persisted BlockMatrix with id $id does not exist."))

  def getPersistedBlockMatrixType(id: String): BlockMatrixType =
    BlockMatrixType.fromBlockMatrix(getPersistedBlockMatrix(id))

  def unpersistBlockMatrix(id: String): Unit = {
    getPersistedBlockMatrix(id).unpersist()
    blockmatrices.remove(id)
  }
}
