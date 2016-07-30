package org.broadinstitute.hail.variant

import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.broadinstitute.hail.check.Gen
import org.json4s._

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
}

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
