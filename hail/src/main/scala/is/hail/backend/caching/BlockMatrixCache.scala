package is.hail.backend.caching

import is.hail.linalg.BlockMatrix

import scala.collection.mutable

class BlockMatrixCache extends mutable.AbstractMap[String, BlockMatrix] with AutoCloseable {

  private[this] val blockmatrices: mutable.Map[String, BlockMatrix] =
    mutable.LinkedHashMap.empty

  override def +=(kv: (String, BlockMatrix)): BlockMatrixCache.this.type = {
    blockmatrices += kv; this
  }

  override def -=(key: String): BlockMatrixCache.this.type = {
    get(key).foreach { bm => bm.unpersist(); blockmatrices -= key }; this
  }

  override def get(key: String): Option[BlockMatrix] =
    blockmatrices.get(key)

  override def iterator: Iterator[(String, BlockMatrix)] =
    blockmatrices.iterator

  override def close(): Unit = {
    blockmatrices.values.foreach(_.unpersist())
    blockmatrices.clear()
  }
}
