package org.broadinstitute.k3.variant

import scala.language.implicitConversions

import org.apache.spark.sql.Row

object RichRow {
  implicit def fromRow(r: Row): RichRow = new RichRow(r)
}

// FIXME switch to to get
class RichRow(r: Row) {
  import RichRow._

  implicit def toVariant: Variant = {
    Variant(r.getAs[String](0),
      r.getAs[Int](1),
      r.getAs[String](2),
      r.getAs[String](3))
  }

  def toGenotypeStream: GenotypeStream = {
    GenotypeStream(r.getAs[Row](0).toVariant,
      r.getAs[Int](1),
      r.getAs[Array[Byte]](2))
  }

  def toVariantGenotypeStreamTuple: (Variant, GenotypeStream) = {
    (r.getAs[Row](0).toVariant,
      r.getAs[Row](1).toGenotypeStream)
  }
}
