package org.broadinstitute.k3.variant

import scala.language.implicitConversions

import org.apache.spark.sql.Row

object RichRow {
  implicit def fromRow(r: Row): RichRow = new RichRow(r)
}

class RichRow(r: Row) {
  import RichRow._

  implicit def getVariant(i: Int): Variant = {
    val ir = r.getAs[Row](i)
    Variant(ir.getAs[String](0),
      ir.getAs[Int](1),
      ir.getAs[String](2),
      ir.getAs[String](3))
  }

  def getGenotypeStream(i: Int): GenotypeStream = {
    val ir = r.getAs[Row](i)
    GenotypeStream(ir.getVariant(0),
      ir.getInt(1),
      ir.getAs[Array[Byte]](2))
  }
}
