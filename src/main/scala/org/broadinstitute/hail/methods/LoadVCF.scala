package org.broadinstitute.hail.methods

import htsjdk.variant.vcf.{VCFHeaderLineCount, VCFHeaderLineType, VCFInfoHeaderLine}
import org.apache.spark.{Accumulable, SparkContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.vcf
import org.broadinstitute.hail.vcf.BufferedLineIterator

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source

case class VCFSettings(storeGQ: Boolean = false,
  skipGenotypes: Boolean = false,
  compress: Boolean = true,
  ppAsPL: Boolean = false,
  skipBadAD: Boolean = false)

object VCFReport {
  val GTPLMismatch = 1
  val ADDPMismatch = 2
  val ODMissingAD = 3
  val ADODDPPMismatch = 4
  val GQPLMismatch = 5
  val GQMissingPL = 6
  val RefNonACGTN = 7
  val Symbolic = 8
  val ADInvalidNumber = 9

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def isVariant(id: Int): Boolean = id == RefNonACGTN || id == Symbolic

  def isGenotype(id: Int): Boolean = !isVariant(id)

  def warningMessage(id: Int, count: Int): String = {
    val desc = id match {
      case GTPLMismatch => "PL(GT) != 0"
      case ADDPMismatch => "sum(AD) > DP"
      case ODMissingAD => "OD present but AD missing"
      case ADODDPPMismatch => "DP != sum(AD) + OD"
      case GQPLMismatch => "GQ != difference of two smallest PL entries"
      case GQMissingPL => "GQ present but PL missing"
      case RefNonACGTN => "REF contains non-ACGT"
      case Symbolic => "Variant is symbolic"
      case ADInvalidNumber => "AD array contained the wrong number of elements"
    }
    s"$count ${plural(count, "time")}: $desc"
  }

  def report() {
    val sb = new StringBuilder()
    for ((file, m) <- accumulators) {
      sb.clear()

      sb.append(s"while importing:\n    $file")

      val variantWarnings = m.value.filter { case (k, v) => isVariant(k) }
      val nVariantsFiltered = variantWarnings.values.sum
      if (nVariantsFiltered > 0) {
        sb.append(s"\n  filtered $nVariantsFiltered variants:")
        variantWarnings.foreach { case (id, n) =>
          if (n > 0) {
            sb.append("\n    ")
            sb.append(warningMessage(id, n))
          }
        }
        warn(sb.result())
      }

      val genotypeWarnings = m.value.filter { case (k, v) => isGenotype(k) }
      val nGenotypesFiltered = genotypeWarnings.values.sum
      if (nGenotypesFiltered > 0) {
        sb.append(s"\n  filtered $nGenotypesFiltered genotypes:")
        genotypeWarnings.foreach { case (id, n) =>
          if (n > 0) {
            sb.append("\n    ")
            sb.append(warningMessage(id, n))
          }
        }
      }

      if (nVariantsFiltered == 0 && nGenotypesFiltered == 0) {
        sb.append("  import clean")
        info(sb.result())
      } else
        warn(sb.result())
    }
  }
}

object LoadVCF {
  def lineRef(s: String): String = {
    var i = 0
    var t = 0
    while (t < 3
      && i < s.length) {
      if (s(i) == '\t')
        t += 1
      i += 1
    }
    val start = i

    while (i < s.length
      && s(i) != '\t')
      i += 1
    val end = i

    s.substring(start, end)
  }

  def infoNumberToString(line: VCFInfoHeaderLine): String = line.getCountType match {
    case VCFHeaderLineCount.A => "A"
    case VCFHeaderLineCount.G => "G"
    case VCFHeaderLineCount.R => "R"
    case VCFHeaderLineCount.INTEGER => line.getCount.toString
    case VCFHeaderLineCount.UNBOUNDED => "."
  }

  def infoTypeToString(line: VCFInfoHeaderLine): String = line.getType match {
    case VCFHeaderLineType.Integer => "Integer"
    case VCFHeaderLineType.Flag => "Flag"
    case VCFHeaderLineType.Float => "Float"
    case VCFHeaderLineType.Character => "Character"
    case VCFHeaderLineType.String => "String"
  }

  def infoField(line: VCFInfoHeaderLine, i: Int): Field = {
    val baseType = line.getType match {
      case VCFHeaderLineType.Integer => TInt
      case VCFHeaderLineType.Float => TDouble
      case VCFHeaderLineType.String => TString
      case VCFHeaderLineType.Character => TChar
      case VCFHeaderLineType.Flag => TBoolean
    }

    val attrs = Map("Description" -> line.getDescription,
      "Number" -> infoNumberToString(line),
      "Type" -> infoTypeToString(line))
    if (line.isFixedCount &&
      (line.getCount == 1 ||
        (line.getType == VCFHeaderLineType.Flag && line.getCount == 0)))
      Field(line.getID, baseType, i, attrs)
    else
      Field(line.getID, TArray(baseType), i, attrs)
  }

  def apply(sc: SparkContext,
    file1: String,
    files: Array[String] = null, // FIXME hack
    storeGQ: Boolean = false,
    compress: Boolean = true,
    nPartitions: Option[Int] = None,
    skipGenotypes: Boolean = false,
    ppAsPL: Boolean = false,
    skipBadAD: Boolean = false): VariantDataset = {

    val settings = VCFSettings(storeGQ, skipGenotypes, compress, ppAsPL, skipBadAD)

    val hConf = sc.hadoopConfiguration
    val headerLines = readFile(file1, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .takeWhile { line => line(0) == '#' }
        .toArray
    }

    val codec = new htsjdk.variant.vcf.VCFCodec()

    val header = codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
      .getHeaderValue
      .asInstanceOf[htsjdk.variant.vcf.VCFHeader]

    // FIXME apply descriptions when HTSJDK is fixed to expose filter descriptions
    val filters: Map[String, String] = header
      .getFilterLines
      .toList
      // (filter, description)
      .map(line => (line.getID, ""))
      .toMap

    val infoHeader = header.getInfoHeaderLines

    val infoSignature = if (infoHeader.size > 0)
      Some(TStruct(infoHeader
        .zipWithIndex
        .map { case (line, i) => infoField(line, i) }
        .toArray))
    else None


    val variantAnnotationSignatures = TStruct(
      Array(
        Some(Field("rsid", TString, 0)),
        Some(Field("qual", TDouble, 1)),
        Some(Field("filters", TSet(TString), 2, filters)),
        Some(Field("pass", TBoolean, 3)),
        infoSignature.map(sig => Field("info", sig, 4))
      ).flatten)

    val headerLine = headerLines.last
    if (!(headerLine(0) == '#' && headerLine(1) != '#'))
      fatal(s"corrupt VCF: expected final header line of format `#CHROM\tPOS\tID...'" +
        s"\n  found: ${truncate(headerLine)}")

    val sampleIds: Array[String] =
      if (skipGenotypes)
        Array.empty
      else
        headerLine
          .split("\t")
          .drop(9)

    val infoSignatureBc = infoSignature.map(sig => sc.broadcast(sig))

    val headerLinesBc = sc.broadcast(headerLines)

    val files2 = if (files == null)
      Array(file1)
    else
      files

    val genotypes = sc.union(files2.map { file =>
      val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
      VCFReport.accumulators ::= (file, reportAcc)

      sc.textFileLines(file, nPartitions.getOrElse(sc.defaultMinPartitions))
        .mapPartitions { lines =>
          val codec = new htsjdk.variant.vcf.VCFCodec()
          val reader = vcf.HtsjdkRecordReader(headerLinesBc.value, codec)
          lines.flatMap { l => l.map { line =>
            if (line.isEmpty || line(0) == '#')
              None
            else if (!lineRef(line).forall(c => c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N')) {
              reportAcc += VCFReport.RefNonACGTN
              None
            } else {
              val vc = codec.decode(line)
              if (vc.isSymbolic) {
                reportAcc += VCFReport.Symbolic
                None
              } else
                Some(reader.readRecord(reportAcc, vc, infoSignatureBc.map(_.value),
                  settings))
            }
          }.value
          }
        }
    })

    VariantSampleMatrix(VariantMetadata(sampleIds,
      Annotation.emptyIndexedSeq(sampleIds.length),
      Annotation.empty,
      TStruct.empty,
      variantAnnotationSignatures,
      TStruct.empty), genotypes)
  }

}
