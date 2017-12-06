package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.{TInt32, TString, TStruct, Type}
import is.hail.sparkextras.OrderedKey
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.json4s._

import scala.reflect.ClassTag
import scala.util.parsing.combinator.JavaTokenParsers
import scala.collection.JavaConverters._
import scala.language.implicitConversions

object LocusImplicits {
  /* We cannot add this to the Locus companion object because it breaks serialization. */
  implicit val orderedKey = new OrderedKey[Locus, Locus] {
    def project(key: Locus): Locus = key

    val kOrd: Ordering[Locus] = Locus.order

    val pkOrd: Ordering[Locus] = Locus.order

    val kct: ClassTag[Locus] = implicitly[ClassTag[Locus]]

    val pkct: ClassTag[Locus] = implicitly[ClassTag[Locus]]
  }
}

object Locus {
  val simpleContigs: Seq[String] = (1 to 22).map(_.toString) ++ Seq("X", "Y", "MT")

  def sparkSchema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("position", IntegerType, nullable = false)))

  def fromRow(r: Row): Locus = {
    Locus(r.getAs[String](0), r.getInt(1))
  }

  def intervalFromRow(r: Row): Interval[Locus] = {
    Interval[Locus](
      Locus.fromRow(r.getAs[Row](0)),
      Locus.fromRow(r.getAs[Row](1))
    )
  }

  def intervalToRow(i: Interval[Locus]): Row =
    Row(i.start.toRow, i.end.toRow)

  def gen(contigs: Seq[String]): Gen[Locus] =
    Gen.zip(Gen.oneOfSeq(contigs), Gen.posInt)
      .map { case (contig, pos) => Locus(contig, pos) }

  def gen: Gen[Locus] = gen(simpleContigs)

  implicit val locusJSONRWer: JSONReaderWriter[Locus] = caseClassJSONReaderWriter[Locus]

  def parse(str: String, gr: GRBase): Locus = {
    str.split(":") match {
      case Array(chr, pos) =>
        val l = Locus(chr, pos.toInt)
        gr.checkLocus(l)
        l
      case a => fatal(s"invalid locus: expected 2 colon-delimited fields, found ${ a.length }: $str")
    }
  }

  object LocusIntervalParser extends JavaTokenParsers {
    def parseInterval(input: String, gr: GRBase): Interval[Locus] = {
      parseAll[Interval[Locus]](interval(gr), input) match {
        case Success(r, _) => r
        case NoSuccess(msg, next) => fatal(s"invalid interval expression: `$input': $msg")
      }
    }

    def interval(gr: GRBase): Parser[Interval[Locus]] = {
      {locus(gr) ~ "-" ~ locus(gr) ^^ { case l1 ~ _ ~ l2 => Interval(l1, l2) } |
        locus(gr) ~ "-" ~ pos ^^ { case l1 ~ _ ~ p2 => Interval(l1, l1.copy(position = p2.getOrElse(gr.contigLength(l1.contig)))) } |
        contig ~ "-" ~ contig ^^ { case c1 ~ _ ~ c2 => Interval(Locus(c1, 1), Locus(c2, gr.contigLength(c2))) } |
        contig ^^ { c => Interval(Locus(c, 1), Locus(c, gr.contigLength(c))) }} ^^ { i =>
        gr.checkInterval(i)
        i
      }
    }

    def locus(gr: GRBase): Parser[Locus] = contig ~ ":" ~ pos ^^ { case c ~ _ ~ p => Locus(c, p.getOrElse(gr.contigLength(c))) }

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

    def pos: Parser[Option[Int]] = {
      "[sS][Tt][Aa][Rr][Tt]".r ^^ { _ => Some(1) } |
        "[Ee][Nn][Dd]".r ^^ { _ => None } |
        "\\d+".r <~ "[Kk]".r ^^ { i => Some(coerceInt(i) * 1000) } |
        "\\d+".r <~ "[Mm]".r ^^ { i => Some(coerceInt(i) * 1000000) } |
        "\\d+".r ~ "." ~ "\\d{1,3}".r ~ "[Kk]".r ^^ { case lft ~ _ ~ rt ~ _ => Some(coerceInt(lft + rt) * exp10(3 - rt.length)) } |
        "\\d+".r ~ "." ~ "\\d{1,6}".r ~ "[Mm]".r ^^ { case lft ~ _ ~ rt ~ _ => Some(coerceInt(lft + rt) * exp10(6 - rt.length)) } |
        "\\d+".r ^^ { i => Some(coerceInt(i)) }
    }
  }

  def parseInterval(str: String, gr: GRBase): Interval[Locus] = LocusIntervalParser.parseInterval(str, gr)

  def parseIntervals(arr: Array[String], gr: GRBase): Array[Interval[Locus]] = arr.map(parseInterval(_, gr))

  def parseIntervals(arr: java.util.ArrayList[String], gr: GRBase): Array[Interval[Locus]] = parseIntervals(arr.asScala.toArray, gr)

  def makeInterval(start: Locus, end: Locus): Interval[Locus] = Interval(start, end)

  implicit val order: Ordering[Locus] = new Ordering[Locus] {
    def compare(x: Locus, y: Locus): Int = x.compare(y)
  }
}

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
