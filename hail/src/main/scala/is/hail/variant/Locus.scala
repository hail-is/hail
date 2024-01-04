package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.expr.Parser
import is.hail.utils._

import org.json4s._

import scala.collection.JavaConverters._
import scala.language.implicitConversions

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object Locus {
  val simpleContigs: Seq[String] = (1 to 22).map(_.toString) ++ Seq("X", "Y", "MT")

  def apply(contig: String, position: Int, rg: ReferenceGenome): Locus = {
    rg.checkLocus(contig, position)
    Locus(contig, position)
  }

  def annotation(contig: String, position: Int, rg: Option[ReferenceGenome]): Annotation =
    rg match {
      case Some(ref) => Locus(contig, position, ref)
      case None => Annotation(contig, position)
    }

  def sparkSchema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("position", IntegerType, nullable = false),
    ))

  def fromRow(r: Row): Locus =
    Locus(r.getAs[String](0), r.getInt(1))

  def gen(rg: ReferenceGenome): Gen[Locus] = for {
    (contig, length) <- Contig.gen(rg)
    pos <- Gen.choose(1, length)
  } yield Locus(contig, pos)

  def parse(str: String, rg: ReferenceGenome): Locus = {
    val elts = str.split(":")
    val size = elts.length
    if (size < 2)
      fatal(s"Invalid string for Locus. Expecting contig:pos -- found '$str'.")

    val contig = elts.take(size - 1).mkString(":")
    Locus(contig, elts(size - 1).toInt, rg)
  }

  def parseInterval(str: String, rg: ReferenceGenome, invalidMissing: Boolean = false): Interval =
    Parser.parseLocusInterval(str, rg, invalidMissing)

  def parseIntervals(arr: Array[String], rg: ReferenceGenome, invalidMissing: Boolean)
    : Array[Interval] = arr.map(parseInterval(_, rg, invalidMissing))

  def parseIntervals(
    arr: java.util.List[String],
    rg: ReferenceGenome,
    invalidMissing: Boolean = false,
  ): Array[Interval] = parseIntervals(arr.asScala.toArray, rg, invalidMissing)

  def makeInterval(
    contig: String,
    start: Int,
    end: Int,
    includesStart: Boolean,
    includesEnd: Boolean,
    rgBase: ReferenceGenome,
    invalidMissing: Boolean = false,
  ): Interval = {
    val rg = rgBase.asInstanceOf[ReferenceGenome]
    rg.toLocusInterval(
      Interval(Locus(contig, start), Locus(contig, end), includesStart, includesEnd),
      invalidMissing,
    )
  }
}

case class Locus(contig: String, position: Int) {
  def toRow: Row = Row(contig, position)

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("position", JInt(position)),
  )

  def copyChecked(rg: ReferenceGenome, contig: String = contig, position: Int = position): Locus = {
    rg.checkLocus(contig, position)
    Locus(contig, position)
  }

  def isAutosomalOrPseudoAutosomal(rg: ReferenceGenome): Boolean =
    isAutosomal(rg) || inXPar(rg) || inYPar(rg)

  def isAutosomal(rg: ReferenceGenome): Boolean = !(inX(rg) || inY(rg) || isMitochondrial(rg))

  def isMitochondrial(rg: ReferenceGenome): Boolean = rg.isMitochondrial(contig)

  def inXPar(rg: ReferenceGenome): Boolean = rg.inXPar(this)

  def inYPar(rg: ReferenceGenome): Boolean = rg.inYPar(this)

  def inXNonPar(rg: ReferenceGenome): Boolean = inX(rg) && !inXPar(rg)

  def inYNonPar(rg: ReferenceGenome): Boolean = inY(rg) && !inYPar(rg)

  private def inX(rg: ReferenceGenome): Boolean = rg.inX(contig)

  private def inY(rg: ReferenceGenome): Boolean = rg.inY(contig)

  override def toString: String = s"$contig:$position"
}
