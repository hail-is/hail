package is.hail.utils.richUtils

import is.hail.utils.ArrayBuilder
import is.hail.variant.Variant
import org.apache.spark.sql.Row

class RichRow(r: Row) {

  def update(i: Int, a: Any): Row = {
    val arr = Array.tabulate(r.size)(r.get)
    arr(i) = a
    Row.fromSeq(arr)
  }

  def getOrIfNull[T](i: Int, t: T): T = {
    if (r.isNullAt(i))
      t
    else
      r.getAs[T](i)
  }

  def getOption(i: Int): Option[Any] = {
    Option(r.get(i))
  }

  def getAsOption[T](i: Int): Option[T] = {
    if (r.isNullAt(i))
      None
    else
      Some(r.getAs[T](i))
  }

  def delete(i: Int): Row = {
    require(i >= 0 && i < r.length)
    new DeleteOneRow(r, i)
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

  def getVariant(i: Int): Variant = Variant.fromRow(r.getAs[Row](i))
}

class DeleteOneRow(parent: Row, deleteIdx: Int) extends Row {
  override def length: Int = parent.length - 1

  override def get(i: Int): Any = if (i < deleteIdx) parent.get(i) else parent.get(i + 1)

  override def copy(): Row = {
    val ab = new ArrayBuilder[Any]()
    (0 until deleteIdx).foreach(ab += parent.get(_))
    (deleteIdx + 1 until parent.length).foreach(ab += parent.get(_))
    val result = ab.result()
    Row.fromSeq(result)
  }
}