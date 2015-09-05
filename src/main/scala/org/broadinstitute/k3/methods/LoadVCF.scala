package org.broadinstitute.k3.methods

import java.net.URI
import java.util
import org.apache.hadoop
import org.apache.hadoop.io.compress.CompressionCodecFactory
import scala.collection.mutable
import scala.io.{Codec, Source}
import org.apache.spark.{SparkConf, SparkContext}
import org.broadinstitute.k3.variant._

object TryOut {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("K3").setMaster("local")
    val sc = new SparkContext(conf)

    // println(LoadVCF(sc, "sparky", "/Users/cseed/sample.vcf").count)
    println(LoadVCF(sc, "sparky", "src/test/resources/gqbydp_test.vcf").count)
  }
}

// ##FORMAT=<ID=GT, ...> is a format field
// GT:AD:... is a format
// 1/0:5,2:... is a genotype

case class VCFFormatField(id: String,
  fields: Map[String, String])

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

case class VCFHeader(lines: Array[String],
  sampleIds: Array[String],
  formatFields: Map[String, VCFFormatField]) {
}

object VCFHeader {
  def apply(lines: Array[String]): VCFHeader = {
    val headerLine = lines.find(line => line.length >= 2 && line(0) == '#' && line(1) != '#').get

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val b = new StringBuilder()

    val formatPrefix = "##FORMAT=<"
    val formatSuffix = ">"

    val formatFields = lines.flatMap(line => {
      if (line.startsWith("##FORMAT")) {
        // FIXME error checking
        val bit = line.iterator.buffered

        def collectWhile(p: (Char) => Boolean): String = {
          b.clear()
          while (p(bit.head))
            b += bit.next()
          b.result()
        }

        def parseValue(): String = {
          if (bit.head == '"') {
            bit.next()
            b.clear()
            while (bit.head != '"')
              b += bit.next()
            bit.next() // "
            b.result()
          } else
            collectWhile(_ != ',')
        }

        def parseBody(): Map[String, String] = {
          if (bit.head == '>')
            Map()
          else {
            if (bit.head == ',')
              bit.next() // ,
            val key = collectWhile(_ != '=')
            bit.next() // =
            val value = parseValue()
            parseBody().updated(key, value)
          }
        }

        def parsePrefix() = {
          while (bit.head != '<')
            bit.next()
          bit.next() // '<'
        }

        def parseFields(): Map[String, String] = {
          parsePrefix()
          parseBody()
        }

        val fields = parseFields()
        Some(VCFFormatField(fields("ID"), fields))
      } else
        None
    })
      .map(format => (format.id, format))
      .toMap

    new VCFHeader(lines, sampleIds, formatFields)
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
  def apply(formatField: VCFFormatField, nAlleles: Int): GenotypeFieldParser[Any] = {
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

object LoadVCF {
  def parseGenotypeStream(vcfHeader: VCFHeader,
    nAlleles: Int,
    format: String,
    bit: BufferedIterator[Char]): Iterator[VCFGenotype] = {

    val fieldParsers: Array[GenotypeFieldParser[Any]] =
      format.split(":").map(id => GenotypeFieldParser(vcfHeader.formatFields(id), nAlleles))

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

  def apply(sc: SparkContext, vsmtype: String, file: String, nPartitions: Option[Int] = None): VariantDataset = {
    require(file.endsWith(".vcf") || file.endsWith(".vcf.bgz")
      // FIXME no
      || file.endsWith(".vcf.gz"))

    val path = new hadoop.fs.Path(file)
    val hadoopConf = sc.hadoopConfiguration
    val fs = hadoop.fs.FileSystem.get(new URI(file), hadoopConf)
    val codecs = new CompressionCodecFactory(hadoopConf)
    val codec = codecs.getCodec(path)
    val rawIS = fs.open(path)
    val decodedIS = if (codec == null)
      rawIS
    else
      codec.createInputStream(rawIS)
    val s = Source.fromInputStream(decodedIS)

    val vcfHeader = VCFHeader(s.getLines()
      .takeWhile(line => line(0) == '#')
      .toArray)
    s.close()

    def parseLine(line: String): Option[(Variant, GenotypeStream)] = {
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
        val vcfGenotypes = parseGenotypeStream(vcfHeader,
          vcfVariant.nAlleles,
          fixed(8),
          line.view.slice(i, line.length).iterator.buffered)

        val variant = vcfVariant.toVariant

        val b = new GenotypeStreamBuilder(variant)
        vcfGenotypes.foreach(g => b += 0 -> g.toGenotype)
        Some((variant, b.result()))
      } else
        None
    }

    val variantRDD = sc.textFile(file, nPartitions.getOrElse(sc.defaultMinPartitions))
      .filter(line => !line.isEmpty && line(0) != '#')
      .flatMap(parseLine)

    // FIXME null should be contig lengths
    VariantSampleMatrix(vsmtype, VariantMetadata(null, vcfHeader.sampleIds, vcfHeader.lines), variantRDD)
  }
}
