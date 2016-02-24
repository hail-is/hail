package org.broadinstitute.hail.variant

import scala.collection.mutable.ArrayBuffer
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

  def toAltAllele: AltAllele = {
    AltAllele(r.getString(0),
      r.getString(1))
  }

  def getVariant(i: Int): Variant = {
    val ir = r.getAs[Row](i)
    Variant(ir.getString(0),
      ir.getInt(1),
      ir.getString(2),
      ir.getAs[ArrayBuffer[Row]](3).map(_.toAltAllele))
  }

  def getGenotype(i: Int): Genotype = throw new UnsupportedOperationException

  def getGenotypeStream(i: Int): GenotypeStream = {
    val ir = r.getAs[Row](i)
    GenotypeStream(ir.getVariant(0),
      if (ir.isNullAt(1)) None else Some(ir.getInt(1)),
      ir.getAs[Array[Byte]](2))
  }

  // FIXME: unify to CallStream
  def getDenseCallStream(i: Int): DenseCallStream = {
    val ir = r.getAs[Row](i)
    DenseCallStream(ir.getByteArray(0), ir.getDouble(1), ir.getDouble(2), ir.getInt(3))
  }

  def getSparseCallStream(i: Int): SparseCallStream = {
    val ir = r.getAs[Row](i)
    SparseCallStream(ir.getByteArray(0), ir.getDouble(1), ir.getDouble(2), ir.getInt(3))
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
