package org.broadinstitute.hail.variant

import scala.collection.mutable
import scala.language.implicitConversions

import org.apache.spark.sql.Row


object RichRow {
  implicit def fromRow(r: Row): RichRow = new RichRow(r)
}

class RichRow(r: Row) {

  import RichRow._

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

  def getOptionAs[T](i: Int): Option[T] = {
    if (r.isNullAt(i))
      None
    else
      Some(r.getAs[T](i))
  }

  def delete(i: Int): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until r.size).foreach { j =>
      if (j != i)
        ab += r.get(j)
    }
    Row.fromSeq(ab.result())
  }

  def append(a: Any): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until r.size).foreach { j =>
      ab += r.get(j)
    }
    ab += a
    Row.fromSeq(ab.result())
  }

  def insertBefore(i: Int, a: Any): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until r.size).foreach { j =>
      if (j == i)
        ab += a
      ab += r.get(j)
    }
    Row.fromSeq(ab.result())
  }

  def getIntOption(i: Int): Option[Int] =
    if (r.isNullAt(i))
      None
    else
      Some(r.getInt(i))

  def toAltAllele: AltAllele = {
    AltAllele(r.getString(0),
      r.getString(1))
  }

  def getVariant(i: Int): Variant = {
    val ir = r.getAs[Row](i)
    Variant(ir.getString(0),
      ir.getInt(1),
      ir.getString(2),
      ir.getAs[mutable.ArrayBuffer[Row]](3).map(_.toAltAllele))
  }

  def getGenotype(i: Int): Genotype = throw new UnsupportedOperationException

  def getGenotypeStream(i: Int): GenotypeStream = {
    val ir = r.getAs[Row](i)
    GenotypeStream(ir.getVariant(0),
      if (ir.isNullAt(1)) None else Some(ir.getInt(1)),
      ir.getAs[Array[Byte]](2))
  }

  def getTuple2String(i: Int): (String, String) = (r.getString(0), r.getString(1))

  def getTuple3String(i: Int): (String, String, String) = (r.getString(0), r.getString(1), r.getString(2))

  def getByteArray(i: Int): Array[Byte] = {
    r.getAs[Array[Byte]](i)
  }
}
