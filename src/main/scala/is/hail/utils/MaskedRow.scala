package is.hail.utils

import org.apache.spark.sql.Row

case class MaskedRow(parent: Row, included: Array[Int]) extends Row {

  def length: Int = included.length

  def get(i: Int): Any = parent.get(included(i))

  def copy() = MaskedRow(parent.copy(), included)
}
