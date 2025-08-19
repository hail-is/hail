package is.hail.utils.richUtils

import is.hail.utils.compat.immutable.ArraySeq

import org.apache.spark.sql.Row

class RichRow(r: Row) {

  def update(i: Int, a: Any): Row = {
    val arr = Array.tabulate(r.size)(r.get)
    arr(i) = a
    Row.fromSeq(ArraySeq.unsafeWrapArray(arr))
  }

  def select(indices: IndexedSeq[Int]): Row = Row.fromSeq(indices.map(r.get))

  def deleteField(i: Int): Row = {
    require(i >= 0 && i < r.length)
    new RowWithDeletedField(r, i)
  }

  def truncate(newSize: Int): Row = {
    require(newSize <= r.size)
    Row.fromSeq(ArraySeq.tabulate(newSize)(i => r.get(i)))
  }
}

class RowWithDeletedField(parent: Row, deleteIdx: Int) extends Row {
  override def length: Int = parent.length - 1

  override def get(i: Int): Any = if (i < deleteIdx) parent.get(i) else parent.get(i + 1)

  override def copy(): Row = this
}
