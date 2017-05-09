package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.{TInt, TString, TStruct, Type}
import is.hail.sparkextras.OrderedKey
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.json4s._

import scala.reflect.ClassTag
import scala.util.parsing.combinator.JavaTokenParsers

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

  def apply(contig: String, position: Int)(implicit gr: GenomeReference): Locus = {
    gr.contigIndex.get(contig) match {
      case Some(idx) => Locus(idx, position)
      case None => fatal(s"Did not find contig `$contig' in genome reference `${ gr.name }'.")
    }
  }

  def sparkSchema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("position", IntegerType, nullable = false)))

  def expandedType: TStruct = TStruct(
    "contig" -> TString,
    "position" -> TInt)

  def gen(gr: GenomeReference): Gen[Locus] =
    Gen.zip(Gen.oneOfSeq(gr.contigs.indices), Gen.posInt)
      .map { case (contig, pos) => Locus(contig, pos) }

  def gen: Gen[Locus] = gen(GenomeReference.GRCh37)

  implicit val locusJSONRWer: JSONReaderWriter[Locus] = caseClassJSONReaderWriter[Locus]

  def parse(str: String, gr: GenomeReference): Locus = {
    str.split(":") match {
      case Array(chr, pos) => Locus(chr, pos.toInt)(gr)
      case a => fatal(s"expected 2 colon-delimited fields, but found ${ a.length }")
    }
  }

  object LocusIntervalParser extends JavaTokenParsers {
    def parseInterval(input: String, gr: GenomeReference): Interval[Locus] = {
      parseAll[Interval[Locus]](interval(gr), input) match {
        case Success(r, _) => r
        case NoSuccess(msg, next) => fatal(s"invalid interval expression: `$input': $msg")
      }
    }

    def interval(gr: GenomeReference): Parser[Interval[Locus]] = {
      locus(gr) ~ "-" ~ locus(gr) ^^ { case l1 ~ _ ~ l2 => Interval(l1, l2) } |
        locus(gr) ~ "-" ~ pos ^^ { case l1 ~ _ ~ p2 => Interval(l1, l1.copy(position = p2)) } |
        contig ~ "-" ~ contig ^^ { case c1 ~ _ ~ c2 => Interval(Locus(c1, 0)(gr), Locus(c2, Int.MaxValue)(gr)) } |
        contig ^^ { c => Interval(Locus(c, 0)(gr), Locus(c, Int.MaxValue)(gr))}
    }

    def locus(gr: GenomeReference): Parser[Locus] = {
      contig ~ ":" ~ pos ^^ { case c ~ _ ~ p => Locus(c, p)(gr) }
    }

    def contig: Parser[String] = "\\w+".r

    def coerceInt(s: String): Int = try {
      s.toInt
    } catch {
      case e: java.lang.NumberFormatException => Int.MaxValue
    }

    def exp10(i: Int): Int = {
      var mult = 1
      var j = 0
      while (j < i) {
        mult *= 10
        j += 1
      }
      mult
    }

    def pos: Parser[Int] = {
      "[sS][Tt][Aa][Rr][Tt]".r ^^ { _ => 0 } |
        "[Ee][Nn][Dd]".r ^^ { _ => Int.MaxValue } |
        "\\d+".r <~ "[Kk]".r ^^ { i => coerceInt(i) * 1000 } |
        "\\d+".r <~ "[Mm]".r ^^ { i => coerceInt(i) * 1000000 } |
        "\\d+".r ~ "." ~ "\\d{1,3}".r ~ "[Kk]".r ^^ { case lft ~ _ ~ rt ~ _ => coerceInt(lft + rt) * exp10(3 - rt.length) } |
        "\\d+".r ~ "." ~ "\\d{1,6}".r ~ "[Mm]".r ^^ { case lft ~ _ ~ rt ~ _ => coerceInt(lft + rt) * exp10(6 - rt.length) } |
        "\\d+".r ^^ { i => coerceInt(i) }
    }
  }

  def parseInterval(str: String)(implicit gr: GenomeReference): Interval[Locus] = LocusIntervalParser.parseInterval(str, gr)

  def parseIntervals(arr: Array[String])(implicit gr: GenomeReference): Array[Interval[Locus]] = arr.map(parseInterval(_)(gr))

  def makeInterval(start: Locus, end: Locus): Interval[Locus] = Interval(start, end)
}

@SerialVersionUID(9197069433877243281L)
case class Locus(contig: Int, position: Int) extends Ordered[Locus] {
  def contigStr(gr: GenomeReference): String = gr.contigNames(contig)

  def compare(that: Locus): Int = {
    val c = contig.compare(that.contig)
    if (c != 0)
      return c

    position.compare(that.position)
  }

  def toRow(gr: GenomeReference): Row = Row(contigStr(gr), position)

  def toJSON(gr: GenomeReference): JValue = JObject(
    ("contig", JString(contigStr(gr))),
    ("position", JInt(position)))

  def toString(gr: GenomeReference): String = s"${ contigStr(gr) }:$position"

  override def toString: String = s"$contig:$position"
}
