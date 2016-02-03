package org.broadinstitute.hail.methods

import org.broadinstitute.hail.vcf.BufferedLineIterator
import scala.io.Source
import org.apache.spark.{Accumulable, SparkContext}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.vcf
import org.broadinstitute.hail.annotations._
import scala.collection.JavaConversions._
import scala.collection.mutable

object VCFReport {
  val GTPLMismatch = 1
  val ADDPMismatch = 2
  val ODMissingAD = 3
  val ADODDPPMismatch = 4

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def warningMessage(id: Int, count: Int): String = id match {
    case GTPLMismatch => s"$count ${plural(count, "time")}: PL(GT) != 0"
    case ADDPMismatch => s"$count ${plural(count, "time")}: sum(AD) > DP"
    case ODMissingAD => s"$count ${plural(count, "time")}: OD present but AD missing"
    case ADODDPPMismatch => s"$count ${plural(count, "time")}: DP != sum(AD) + OD"
  }

  def report() {
    val sb = new StringBuilder()
    for ((file, m) <- accumulators) {
      sb.clear()
      val nFiltered = m.value.values.sum
      if (nFiltered > 0) {
        sb.append(s"filtered $nFiltered genotypes while importing:\n  $file\ndetails:")
        m.value.foreach { case (id, n) =>
          if (n > 0) {
            sb += '\n'
            sb.append("    ")
            sb.append(warningMessage(id, n))
          }
        }
        warning(sb.result())
      } else {
        sb.append(s"import clean while importing:\n  $file")
        info(sb.result())
      }
    }
  }
}

object LoadVCF {
  def apply(sc: SparkContext,
    file: String,
    compress: Boolean = true,
    nPartitions: Option[Int] = None): VariantDataset = {

    require(file.endsWith(".vcf")
      || file.endsWith(".vcf.bgz")
      || file.endsWith(".vcf.gz"))

    val hConf = sc.hadoopConfiguration
    val headerLines = readFile(file, hConf) { s =>
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
    val filters: IndexedSeq[(String, String)] = header
      .getFilterLines
      .toList
      .map(line => (line.getID, ""))
      .toArray[(String, String)]

    val infoSignatures = Annotations(header
      .getInfoHeaderLines
      .toList
      .map(line => (line.getID, VCFSignature.parse(line)))
      .toMap)

    val variantAnnotationSignatures: Annotations = Annotations(Map("info" -> infoSignatures,
      "filters" -> new SimpleSignature("Set[String]"),
      "pass" -> new SimpleSignature("Boolean"),
      "qual" -> new SimpleSignature("Double"),
      "multiallelic" -> new SimpleSignature("Boolean"),
      "rsid" -> new SimpleSignature("String")))

    val headerLine = headerLines.last
    assert(headerLine(0) == '#' && headerLine(1) != '#')

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val sigMap = sc.broadcast(infoSignatures.attrs)

    val headerLinesBc = sc.broadcast(headerLines)

    val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
    VCFReport.accumulators ::= (file, reportAcc)

    val genotypes = sc.textFile(file, nPartitions.getOrElse(sc.defaultMinPartitions))
      .mapPartitions { lines =>
        val reader = vcf.HtsjdkRecordReader(headerLinesBc.value)
        lines.filter(line => !line.isEmpty && line(0) != '#')
          .map(line => reader.readRecord(reportAcc, line, sigMap.value))
      }

    VariantSampleMatrix(VariantMetadata(filters, sampleIds,
      Annotations.emptyIndexedSeq(sampleIds.length), Annotations.empty(),
      variantAnnotationSignatures), genotypes)
  }
}
