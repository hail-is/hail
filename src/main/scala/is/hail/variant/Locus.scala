package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.{TInt, TString, TStruct, Type}
import is.hail.sparkextras.OrderedKey
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.json4s._

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

  val t: Type = TStruct(
    "contig" -> TString,
    "position" -> TInt)

  def gen(contigs: Seq[String]): Gen[Locus] =
    Gen.zip(Gen.oneOfSeq(contigs), Gen.posInt)
      .map { case (contig, pos) => Locus(contig, pos) }

  def gen: Gen[Locus] = gen(simpleContigs)

  implicit val locusJSONRWer: JSONReaderWriter[Locus] = caseClassJSONReaderWriter[Locus]

  def parse(str: String): Locus = {
    str.split(":") match {
      case Array(chr, pos) => Locus(chr, pos.toInt)
      case a => fatal(s"expected 2 colon-delimited fields, but found ${ a.length }")
    }
  }

  def parseInterval(str: String): Interval[Locus] = {
    str.split("-") match {
      case Array(start, end) =>
        val startLocus = Locus.parse(start)
        val endLocus = end.split(":") match {
          case Array(pos) => Locus(startLocus.contig, pos.toInt)
          case Array(chr, pos) => Locus(chr, pos.toInt)
          case a => fatal(s"expected end locus in format CHR:POS or POS, but found ${a.length} colon-delimited fields")
        }
        Interval(startLocus, endLocus)
      case a => fatal(s"expected 2 dash-delimited fields, but found ${a.length}")
    }
  }

  def makeInterval(start: Locus, end: Locus): Interval[Locus] = Interval(start, end)
}

@SerialVersionUID(9197069433877243281L)
case class Locus(contig: String, position: Int) extends Ordered[Locus] {
  def compare(that: Locus): Int = {
    var c = Contig.compare(contig, that.contig)
    if (c != 0)
      return c

    position.compare(that.position)
  }

  def toRow: Row = Row(contig, position)

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("position", JInt(position)))

  override def toString: String = s"$contig:$position"
}
