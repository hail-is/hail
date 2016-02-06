package org.broadinstitute.hail.variant

import scala.language.implicitConversions

import org.apache.spark.sql.Row

object RichRow {
  implicit def fromRow(r: Row): RichRow = new RichRow(r)
}

class RichRow(r: Row) {

  import RichRow._

  def getVariant(i: Int): Variant = {
    val ir = r.getAs[Row](i)
    Variant(ir.getAs[String](0),
      ir.getAs[Int](1),
      ir.getAs[String](2),
      ir.getAs[String](3))
  }

  def getGenotype(i: Int): Genotype = {
    val ir = r.getAs[Row](i)
    val i1r = ir.getAs[Row](1)
    val i3r = ir.getAs[Row](3)

    Genotype(ir.getInt(0),
      (i1r.getInt(0),
        i1r.getInt(1)),
      ir.getInt(2),
      if (i3r != null)
        (i3r.getInt(0),
          i3r.getInt(1),
          i3r.getInt(2))
      else
        null)
  }

  def getGenotypeStream(i: Int): GenotypeStream = {
    val ir = r.getAs[Row](i)
    GenotypeStream(ir.getVariant(0),
      if (ir.isNullAt(1)) None else Some(ir.getInt(1)),
      ir.getAs[Array[Byte]](2))
  }

  def getDenseCallStream(i: Int): DenseCallStream = {
    val ir = r.getAs[Row](i)
    DenseCallStream(ir.getByteArray(0), ir.getDouble(1), ir.getDouble(2), ir.getInt(3))
  }

  def getTuple2String(i: Int): (String, String) = (r.getString(0), r.getString(1))
  def getTuple3String(i: Int): (String, String, String) = (r.getString(0), r.getString(1), r.getString(2))

  def getByteArray(i: Int): Array[Byte] = {
    r.getAs[Array[Byte]](i)
  }

  def getIntArray(i: Int): Array[Int] = {
    r.getAs[Array[Int]](i)
  }
}
