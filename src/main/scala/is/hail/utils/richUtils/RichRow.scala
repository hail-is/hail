package is.hail.utils.richUtils

import is.hail.utils.ArrayBuilder
import org.apache.spark.sql.Row

class RichRow(r: Row) {

  def update(i: Int, a: Any): Row = {
    val arr = Array.tabulate(r.size)(r.get)
    arr(i) = a
    Row.fromSeq(arr)
  }

  def deleteField(i: Int): Row = {
    require(i >= 0 && i < r.length)
    new RowWithDeletedField(r, i)
  }

  def append(a: Any): Row = {
    val ab = new ArrayBuilder[Any]()
    ab ++= r.toSeq
    ab += a
    Row.fromSeq(ab.result())
  }

  def insertBefore(i: Int, a: Any): Row = {
    val ab = new ArrayBuilder[Any]()
    (0 until i).foreach(ab += r.get(_))
    ab += a
    (i until r.size).foreach(ab += r.get(_))
    Row.fromSeq(ab.result())
  }
}

class RowWithDeletedField(parent: Row, deleteIdx: Int) extends Row {
  override def length: Int = parent.length - 1

  override def get(i: Int): Any = if (i < deleteIdx) parent.get(i) else parent.get(i + 1)

  override def copy(): Row = this
}