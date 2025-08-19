package is.hail.sparkextras.implicits

import is.hail.collection.compat.immutable.ArraySeq

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

  def iterator: Iterator[Any] =
    new Iterator[Any] {
      private[this] var idx: Int = 0
      override def hasNext: Boolean = idx < r.size

      override def next(): Any = {
        val a = r(idx)
        idx += 1
        a
      }
    }
}

class RowWithDeletedField(parent: Row, deleteIdx: Int) extends Row {
  override def length: Int = parent.length - 1

  override def get(i: Int): Any = if (i < deleteIdx) parent.get(i) else parent.get(i + 1)

  override def copy(): Row = this
}
