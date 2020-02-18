package is.hail.backend.spark

import is.hail.backend.ValueCache
import is.hail.linalg.BlockMatrix

import scala.collection.mutable

case class SparkValueCache() extends ValueCache {
  private[this] val blockmatrices: mutable.Map[String, BlockMatrix] = new mutable.HashMap()

  def persistBlockMatrix(id: String, value: BlockMatrix, storageLevel: String): Unit =
    blockmatrices.update(id, value.persist(storageLevel))

  def getPersistedBlockMatrix(id: String): BlockMatrix = blockmatrices(id)

  def unpersistBlockMatrix(id: String): Unit =
    blockmatrices(id).unpersist()
}
