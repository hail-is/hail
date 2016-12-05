package org.broadinstitute.hail.variant

import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.sparkextras.OrderedKey
import org.broadinstitute.hail.utils._
import org.json4s._
import org.json4s.Extraction.decompose
import org.json4s.jackson.Serialization

import scala.reflect.ClassTag

object LocusImplicits {
  /* We cannot add this to the Locus companion object because it breaks serialization. */
  implicit val orderedKey = new OrderedKey[Locus, Locus] {
    def project(key: Locus): Locus = key

    def kOrd: Ordering[Locus] = implicitly[Ordering[Locus]]

    def pkOrd: Ordering[Locus] = implicitly[Ordering[Locus]]

    def kct: ClassTag[Locus] = implicitly[ClassTag[Locus]]

    def pkct: ClassTag[Locus] = implicitly[ClassTag[Locus]]
  }
}

object Locus {
  val simpleContigs: Seq[String] = (1 to 22).map(_.toString) ++ Seq("X", "Y", "MT")

  val schema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("position", IntegerType, nullable = false)))

  def gen(contigs: Seq[String]): Gen[Locus] =
    Gen.zip(Gen.oneOfSeq(contigs), Gen.posInt)
      .map { case (contig, pos) => Locus(contig, pos) }

  def gen: Gen[Locus] = gen(simpleContigs)

  implicit val locusJSONRWer: JSONReaderWriter[Locus] = caseClassJSONReaderWriter[Locus]
}

@SerialVersionUID(9197069433877243281L)
case class Locus(contig: String, position: Int) extends Ordered[Locus] {
  def compare(that: Locus): Int = {
    var c = Contig.compare(contig, that.contig)
    if (c != 0)
      return c

    position.compare(that.position)
  }

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("position", JInt(position)))

  override def toString: String = s"$contig:$position"
}
