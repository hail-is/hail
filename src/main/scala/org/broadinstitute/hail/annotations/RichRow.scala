package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.language.implicitConversions


object RichRow {
  implicit def fromRow(r: Row): RichRow = new RichRow(r)
}

class RichRow(r: Row) {

  def update(i: Int, a: Any): Row = {
    val arr = r.toSeq.toArray
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

  def getOptionAs[T](i: Int): Option[T] = {
    if (r.isNullAt(i))
      None
    else
      Some(r.getAs[T](i))
  }

  def delete(i: Int): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until i).foreach(ab += r.get(_))
    (i + 1 until r.size).foreach(ab += r.get(_))
    Row.fromSeq(ab.result())
  }

  def append(a: Any): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    ab ++= r.toSeq
    ab += a
    Row.fromSeq(ab.result())
  }

  def insertBefore(i: Int, a: Any): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until i).foreach(ab += r.get(_))
    ab += a
    (i until r.size).foreach(ab += r.get(_))
    Row.fromSeq(ab.result())
  }
}
