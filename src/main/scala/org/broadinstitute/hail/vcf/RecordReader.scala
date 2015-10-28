package org.broadinstitute.hail.vcf

import java.util
import org.broadinstitute.hail.variant._

case class VCFVariant(contig: String,
  start: Int,
  alleles: IndexedSeq[String]) {

  def nAlleles = alleles.size

  def toVariant: Variant = {
    // FIXME
    assert(alleles.size == 2)
    Variant(contig, start, alleles(0), alleles(1))
  }
}

case class VCFGenotype(gt: Option[Array[Int]] = None,
  ad: Option[Array[Int]] = None,
  dp: Option[Int] = None,
  gq: Option[Int] = None,
  pl: Option[Array[Int]] = None,
  other: Map[String, Any] = Map()) {

  override def toString: String = {
    (Array(
      gt.map(util.Arrays.toString),
      ad.map(util.Arrays.toString),
      dp.map(_.toString),
      gq.map(_.toString),
      pl.map(util.Arrays.toString))
      .map(_.getOrElse(".")) ++ other.values.map(_.toString))
      .mkString(":")
  }

  // FIXME this makes me want to puke
  def toGenotype: Genotype = {
    // FIXME
    val gt2 = gt match {
      case Some(a) =>
        assert(a.size == 2)
        (if (a(0) != 0) 1 else 0) +
          (if (a(1) != 0) 1 else 0)
      case None => -1
    }

    val ad2 = ad match {
      case Some(a) =>
        assert(a.size == 2)
        (a(0), a(1))
      case None => (0, 0)
    }

    val dp2 = dp.getOrElse(0) max (ad2._1 + ad2._2)

    val pl2 = pl match {
      case Some(a) =>
        assert(a.size == 3)
        (a(0), a(1), a(2))
      case None =>
        null
    }

    Genotype(gt2, ad2, dp2, gt2 match {
      case -1 => null
      case 0 => (0, pl2._2, pl2._3)
      case 1 => (pl2._1, 0, pl2._3)
      case 2 => (pl2._1, pl2._2, 0)
    })
  }
}

class GenotypeFieldParser[+T](parsef: String => Option[T],
  update: (VCFGenotype, T) => VCFGenotype) {
  val b = new StringBuilder()

  // FIXME don't collect field
  def collectField(bit: BufferedIterator[Char]): Option[String] = {
    if (!bit.hasNext || bit.head == '\t' || bit.head == ':')
      None
    else {
      b.clear()
      while (bit.hasNext && bit.head != '\t' && bit.head != ':')
        b += bit.next()
      Some(b.result())
    }
  }

  def parse(bit: BufferedIterator[Char], g: VCFGenotype): VCFGenotype =
    collectField(bit).flatMap(parsef).map(v => update(g, v)).getOrElse(g)
}

object GenotypeFieldParser {
  def apply(formatField: FormatField, nAlleles: Int): GenotypeFieldParser[Any] = {
    val id = formatField.id
    val numberStr = formatField.fields("Number")
    val typeStr = formatField.fields("Type")

    def parseGT(s: String): Option[Array[Int]] = {
      val alleles = s.split("[|/]").map(a => if (a == ".") None else Some(a.toInt))
      // FIXME check agreement
      if (alleles(0).isDefined)
        Some(alleles.map(_.get))
      else
        None
    }

    def parseInt(s: String): Option[Int] = if (s == ".")
      None
    else
      Some(s.toInt)

    def parseArrayInt(s: String): Option[Array[Int]] = if (s == ".")
      None
    else
      Some(s.split(",").map(_.toInt))

    // FIXME warning if number, type not as expected
    id match {
      case "GT" =>
        assert(numberStr == "1" && typeStr == "String")
        new GenotypeFieldParser[Array[Int]](parseGT,
          (g, gt) => g.copy(gt = Some(gt)))
      case "AD" =>
        assert((numberStr == "R" || numberStr == ".") && typeStr == "Integer")
        new GenotypeFieldParser[Array[Int]](parseArrayInt,
          (g, ad) => g.copy(ad = Some(ad)))
      case "DP" =>
        assert(numberStr == "1" && typeStr == "Integer")
        new GenotypeFieldParser[Int](parseInt,
          (g, dp) => g.copy(dp = Some(dp)))
      case "GQ" =>
        assert(numberStr == "1" && typeStr == "Integer")
        new GenotypeFieldParser[Int](parseInt,
          (g, gq) => g.copy(gq = Some(gq)))
      case "PL" =>
        assert(numberStr == "G" && typeStr == "Integer")
        new GenotypeFieldParser[Array[Int]](parseArrayInt,
          (g, pl) => g.copy(pl = Some(pl)))

      case _ =>
        val subfieldParser: String => Any = typeStr match {
          case "Integer" => _.toInt
          case "Float" => _.toDouble
          case "Flag" => _ => true
          case "Character" => _(0)
          case "String" => s => s
        }

        val (number, arrayField) = numberStr match {
          case "A" => (Some(nAlleles - 1), true)
          case "R" => (Some(nAlleles), true)
          case "G" => (Some((nAlleles + 1) * nAlleles / 2), true)
          case "." => (None, true)
          case s: String =>
            val m = s.toInt
            if (m == 1)
              (None, false)
            else
              (Some(m), true)
        }

        val fieldParser: String => Option[Any] = if (arrayField) {
          (s: String) => if (s == ".")
            None
          else
          // FIXME check number
            Some(s.split(",").map(subfieldParser))
        } else {
          (s: String) => if (s == ".")
            None
          else
            Some(subfieldParser(s))
        }

        new GenotypeFieldParser[Any](fieldParser,
          (g, v) => g.copy(other = g.other.updated(id, v)))
    }
  }
}

class RecordReader(header: Header) extends AbstractRecordReader {
  def parseGenotypeStream(header: Header,
    nAlleles: Int,
    format: String,
    bit: BufferedIterator[Char]): Iterator[VCFGenotype] = {

    val fieldParsers: Array[GenotypeFieldParser[Any]] =
      format.split(":").map(id => GenotypeFieldParser(header.formatFields(id), nAlleles))

    def parseGenotype(): VCFGenotype =
      fieldParsers.foldLeft(VCFGenotype())((g, gfp) => {
        val newg = gfp.parse(bit, g)
        if (bit.hasNext && bit.head != '\t') {
          if (bit.next() != ':')
          // FIXME invalid input exception
            throw new RuntimeException("expected `:'")
        }
        newg
      })

    new Iterator[VCFGenotype] {
      def hasNext = bit.hasNext

      def next(): VCFGenotype = {
        val g = parseGenotype()
        if (bit.hasNext && bit.head == '\t')
          bit.next()
        g
      }
    }
  }

  override def readRecord(line: String): Option[(Variant, Iterator[Genotype])] = {
    var i = 0
    var tabs = 0
    while (tabs < 9) {
      if (line(i) == '\t')
        tabs += 1
      i += 1
    }

    val fixed: Array[String] = line.slice(0, i).split("\t")

    val vcfVariant = VCFVariant(fixed(0),
      fixed(1).toInt,
      fixed(3) +: fixed(4).split(","))

    // FIXME skip multi-allelic variants
    if (vcfVariant.nAlleles == 2) {
      val vcfGenotypes = parseGenotypeStream(header,
        vcfVariant.nAlleles,
        fixed(8),
        line.view.slice(i, line.length).iterator.buffered)

      val variant = vcfVariant.toVariant

      Some((variant,
        for (g <- vcfGenotypes)
          yield g.toGenotype))
    } else
      None
  }
}

object RecordReaderBuilder extends AbstractRecordReaderBuilder {
  def result(headerLines: Array[String]): RecordReader = {
    new RecordReader(Header(headerLines))
  }
}
