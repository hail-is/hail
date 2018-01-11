package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.types.TInterval
import is.hail.expr.Parser
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.json4s._

import scala.collection.JavaConverters._
import scala.language.implicitConversions

object Locus {
  val simpleContigs: Seq[String] = (1 to 22).map(_.toString) ++ Seq("X", "Y", "MT")

  def apply(contig: String, position: Int, gr: GRBase): Locus = {
    gr.checkLocus(contig, position)
    Locus(contig, position)
  }

  def sparkSchema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("position", IntegerType, nullable = false)))

  def fromRow(r: Row): Locus = {
    Locus(r.getAs[String](0), r.getInt(1))
  }
  
  def gen(contigs: Seq[String]): Gen[Locus] =
    Gen.zip(Gen.oneOfSeq(contigs), Gen.posInt)
      .map { case (contig, pos) => Locus(contig, pos) }

  def gen: Gen[Locus] = gen(simpleContigs)

  def gen(gr: GenomeReference): Gen[Locus] = for {
    (contig, length) <- Contig.gen(gr)
    pos <- Gen.choose(1, length)
  } yield Locus(contig, pos)

  def gen(contig: String, length: Int): Gen[Locus] = for {
    pos <- Gen.choose(1, length)
  } yield Locus(contig, pos)

  implicit val locusJSONRWer: JSONReaderWriter[Locus] = caseClassJSONReaderWriter[Locus]

  def parse(str: String, gr: GRBase): Locus = {
    val elts = str.split(":")
    val size = elts.length
    if (size < 2)
      fatal(s"Invalid string for Locus. Expecting contig:pos -- found `$str'.")

    val contig = elts.take(size - 1).mkString(":")
    Locus(contig, elts(size - 1).toInt, gr)
  }

  def parseInterval(str: String, gr: GRBase): Interval = Parser.parseLocusInterval(str, gr)

  def parseIntervals(arr: Array[String], gr: GRBase): Array[Interval] = arr.map(parseInterval(_, gr))

  def parseIntervals(arr: java.util.ArrayList[String], gr: GRBase): Array[Interval] = parseIntervals(arr.asScala.toArray, gr)

  def makeInterval(start: Locus, end: Locus, gr: GRBase): Interval = {
    gr.checkInterval(start, end)
    Interval(start, end)
  }

  def makeInterval(contig: String, start: Int, end: Int, gr: GRBase): Interval = {
    gr.checkInterval(contig, start, end)
    Interval(Locus(contig, start), Locus(contig, end))
  }
}

case class Locus(contig: String, position: Int) {
  def compare(that: Locus, gr: GenomeReference): Int = gr.compare(this, that)

  def toRow: Row = Row(contig, position)

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("position", JInt(position)))

  def copyChecked(gr: GRBase, contig: String = contig, position: Int = position): Locus = {
    gr.checkLocus(contig, position)
    Locus(contig, position)
  }

  override def toString: String = s"$contig:$position"
}
