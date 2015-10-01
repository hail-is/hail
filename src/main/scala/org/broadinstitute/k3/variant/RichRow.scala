package org.broadinstitute.k3.variant

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
}
