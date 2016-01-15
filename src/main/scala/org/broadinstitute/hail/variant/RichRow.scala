package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._
import scala.collection.mutable

import scala.collection.mutable.ArrayBuffer
import scala.language.implicitConversions
import com.esotericsoftware.kryo

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

  def getTuple2String(i: Int): (String, String) = (r.getString(0), r.getString(1))
  def getTuple3String(i: Int): (String, String, String) = (r.getString(0), r.getString(1), r.getString(2))

  def getVariantAnnotations(i: Int): AnnotationData = {

    val ir = r.getAs[Row](i)
    val mapR = ir.getAs[mutable.WrappedArray[Row]](0)
    val valR = ir.getAs[mutable.WrappedArray[Row]](1)
  org.broadinstitute.hail.Utils
    Annotations.fromIndexedSeqs[String]((0 until mapR.length).map(i => mapR(i).getTuple3String(i)),
      (0 until valR.length).map(i => valR(i).getTuple2String(i)))
  }
}
