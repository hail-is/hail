package org.broadinstitute.hail.variant

import scala.language.implicitConversions

import org.apache.spark.sql.Row

object RichRow {
  implicit def fromRow(r: Row): RichRow = new RichRow(r)
}

class RichRow(r: Row) {

  import RichRow._

  def getIntOption(i: Int): Option[Int] =
    if (r.isNullAt(i))
      None
    else
      Some(r.getInt(i))

  def getVariant(i: Int): Variant = throw new UnsupportedOperationException
  /*
  def getVariant(i: Int): Variant = {
    val ir = r.getAs[Row](i)
    Variant(ir.getAs[String](0),
      ir.getAs[Int](1),
      ir.getAs[String](2),
      ir.getAs[String](3))
  }
  */

  def getGenotype(i: Int): Genotype = throw new UnsupportedOperationException

  def getGenotypeStream(i: Int): GenotypeStream = {
    val ir = r.getAs[Row](i)
    GenotypeStream(ir.getVariant(0),
      if (ir.isNullAt(1)) None else Some(ir.getInt(1)),
      ir.getAs[Array[Byte]](2))
  }
}
