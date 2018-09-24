package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.expr.Parser
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.json4s._

import scala.collection.JavaConverters._
import scala.language.implicitConversions

object Locus {
  val simpleContigs: Seq[String] = (1 to 22).map(_.toString) ++ Seq("X", "Y", "MT")

  def apply(contig: String, position: Int, rg: RGBase): Locus = {
    rg.checkLocus(contig, position)
    Locus(contig, position)
  }

  def annotation(contig: String, position: Int, rg: Option[ReferenceGenome]): Annotation = {
    rg match {
      case Some(ref) => Locus(contig, position, ref)
      case None => Annotation(contig, position)
    }
  }

  def sparkSchema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("position", IntegerType, nullable = false)))

  def fromRow(r: Row): Locus = {
    Locus(r.getAs[String](0), r.getInt(1))
  }

  def gen(rg: ReferenceGenome): Gen[Locus] = for {
    (contig, length) <- Contig.gen(rg)
    pos <- Gen.choose(1, length)
  } yield Locus(contig, pos)

  def parse(str: String, rg: RGBase): Locus = {
    val elts = str.split(":")
    val size = elts.length
    if (size < 2)
      fatal(s"Invalid string for Locus. Expecting contig:pos -- found `$str'.")

    val contig = elts.take(size - 1).mkString(":")
    Locus(contig, elts(size - 1).toInt, rg)
  }

  def parseInterval(str: String, rg: RGBase): Interval = Parser.parseLocusInterval(str, rg)

  def parseIntervals(arr: Array[String], rg: RGBase): Array[Interval] = arr.map(parseInterval(_, rg))

  def parseIntervals(arr: java.util.ArrayList[String], rg: RGBase): Array[Interval] = parseIntervals(arr.asScala.toArray, rg)

  def makeInterval(contig: String, start: Int, end: Int, includesStart: Boolean, includesEnd: Boolean, rgBase: RGBase): Interval = {
    val rg = rgBase.asInstanceOf[ReferenceGenome]
    val i = rg.normalizeLocusInterval(Interval(Locus(contig, start), Locus(contig, end), includesStart, includesEnd))
    rg.checkLocusInterval(i)
    i
  }
}

case class Locus(contig: String, position: Int) {
  def compare(that: Locus, rg: ReferenceGenome): Int = rg.compare(this, that)

  def toRow: Row = Row(contig, position)

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("position", JInt(position)))

  def copyChecked(rg: RGBase, contig: String = contig, position: Int = position): Locus = {
    rg.checkLocus(contig, position)
    Locus(contig, position)
  }

  def isAutosomalOrPseudoAutosomal(rg: RGBase): Boolean = isAutosomal(rg) || inXPar(rg) || inYPar(rg)

  def isAutosomal(rg: RGBase): Boolean = !(inX(rg) || inY(rg) || isMitochondrial(rg))

  def isMitochondrial(rg: RGBase): Boolean = rg.isMitochondrial(contig)

  def inXPar(rg: RGBase): Boolean = rg.inXPar(this)

  def inYPar(rg: RGBase): Boolean = rg.inYPar(this)

  def inXNonPar(rg: RGBase): Boolean = inX(rg) && !inXPar(rg)

  def inYNonPar(rg: RGBase): Boolean = inY(rg) && !inYPar(rg)

  private def inX(rg: RGBase): Boolean = rg.inX(contig)

  private def inY(rg: RGBase): Boolean = rg.inY(contig)

  override def toString: String = s"$contig:$position"
}
