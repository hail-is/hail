package is.hail.utils

import org.apache.spark.sql.Row

class MutableRow(val values: Array[Any]) extends Row {

  protected def this() = this(null)

  def this(size: Int) = this(new Array[Any](size))

  override def length: Int = values.length

  override def get(i: Int): Any = values(i)

  override def toSeq: Seq[Any] = values: Seq[Any]

  override def copy(): Row = this

  def update(i: Int, a: Any) {
    values(i) = a
  }
}

object MutableRow {
  def ofSize(i: Int): MutableRow = new MutableRow(Array.fill[Any](i)(null))

  def apply(args: Any*): MutableRow = new MutableRow(args.toArray)

  def fromSeq(args: Seq[Any]): MutableRow = new MutableRow(args.toArray)
}