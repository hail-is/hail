package is.hail.utils.richUtils

import scala.collection.mutable

import org.apache.spark.sql.Row

class RichRow(r: Row) {

  def update(i: Int, a: Any): Row = {
    val arr = Array.tabulate(r.size)(r.get)
    arr(i) = a
    Row.fromSeq(arr)
  }

  def select(indices: Array[Int]): Row = Row.fromSeq(indices.map(r.get))

  def deleteField(i: Int): Row = {
    require(i >= 0 && i < r.length)
    new RowWithDeletedField(r, i)
  }

  def append(a: Any): Row = {
    val ab = new mutable.ArrayBuffer[Any]()
    ab ++= r.toSeq
    ab += a
    Row.fromSeq(ab)
  }

  def insertBefore(i: Int, a: Any): Row = {
    val ab = new mutable.ArrayBuffer[Any]()
    (0 until i).foreach(ab += r.get(_))
    ab += a
    (i until r.size).foreach(ab += r.get(_))
    Row.fromSeq(ab)
  }

  def truncate(newSize: Int): Row = {
    require(newSize <= r.size)
    Row.fromSeq(Array.tabulate(newSize)(i => r.get(i)))
  }
}

class RowWithDeletedField(parent: Row, deleteIdx: Int) extends Row {
  override def length: Int = parent.length - 1

  override def get(i: Int): Any = if (i < deleteIdx) parent.get(i) else parent.get(i + 1)

  override def copy(): Row = this
}
