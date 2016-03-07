package org.broadinstitute.hail.methods

import htsjdk.tribble.TribbleException
import org.broadinstitute.hail.vcf.BufferedLineIterator
import scala.io.Source
import org.apache.spark.{Accumulable, SparkContext}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.{expr, PropagatedTribbleException, vcf}
import org.broadinstitute.hail.annotations._
import scala.collection.JavaConversions._
import scala.collection.mutable

object VCFReport {
  val GTPLMismatch = 1
  val ADDPMismatch = 2
  val ODMissingAD = 3
  val ADODDPPMismatch = 4
  val GQPLMismatch = 5
  val GQMissingPL = 6

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def warningMessage(id: Int, count: Int): String = {
    val desc = id match {
      case GTPLMismatch => "PL(GT) != 0"
      case ADDPMismatch => "sum(AD) > DP"
      case ODMissingAD => "OD present but AD missing"
      case ADODDPPMismatch => "DP != sum(AD) + OD"
      case GQPLMismatch => "GQ != difference of two smallest PL entries"
      case GQMissingPL => "GQ present but PL missing"
    }
    s"$count ${plural(count, "time")}: $desc"
  }

  def report() {
    val sb = new StringBuilder()
    for ((file, m) <- accumulators) {
      sb.clear()
      val nFiltered = m.value.values.sum
      if (nFiltered > 0) {
        sb.append(s"filtered $nFiltered genotypes while importing:\n    $file\n  details:")
        m.value.foreach { case (id, n) =>
          if (n > 0) {
            sb += '\n'
            sb.append("    ")
            sb.append(warningMessage(id, n))
          }
        }
        warn(sb.result())
      } else {
        sb.append(s"import clean while importing:\n  $file")
        info(sb.result())
      }
    }
  }
}

object LoadVCF {
  def apply(sc: SparkContext,
    file1: String,
    files: Array[String] = null, // FIXME hack
    storeGQ: Boolean = false,
    compress: Boolean = true,
    nPartitions: Option[Int] = None): VariantDataset = {

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
    val filters: IndexedSeq[(String, String)] = header
      .getFilterLines
      .toList
      .map(line => (line.getID, ""))
      .toArray[(String, String)]



    val infoRowSignatures = header
      .getInfoHeaderLines
      .toList
      .zipWithIndex
      .map { case (line, index) => (line.getID, VCFSignature.parse(line)) }
      .toArray

    val variantAnnotationSignatures: StructSignature = StructSignature(Map(
      "qual" ->(0, SimpleSignature(expr.TDouble)),
      "filters" ->(1, SimpleSignature(expr.TSet(expr.TString))),
      "pass" ->(2, SimpleSignature(expr.TBoolean)),
      "rsid" ->(3, SimpleSignature(expr.TString)),
      "info" ->(4, StructSignature(infoRowSignatures.zipWithIndex
        .map { case ((key, sig), index) => (key, (index, sig)) }
        .toMap))))

    val headerLine = headerLines.last
    assert(headerLine(0) == '#' && headerLine(1) != '#')

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val infoRowSigsBc = sc.broadcast(infoRowSignatures)

    val headerLinesBc = sc.broadcast(headerLines)

    val files2 = if (files == null)
      Array(file1)
    else
      files

    val genotypes = sc.union(files2.map { file =>
      val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
      VCFReport.accumulators ::=(file, reportAcc)

      sc.textFile(file, nPartitions.getOrElse(sc.defaultMinPartitions))
        .mapPartitions { lines =>
          val reader = vcf.HtsjdkRecordReader(headerLinesBc.value)
          lines.filter(line => !line.isEmpty && line(0) != '#')
            .map { line =>
              try {
                reader.readRecord(reportAcc, line, infoRowSigsBc.value, storeGQ)
              } catch {
                case e: TribbleException =>
                  log.error(s"${e.getMessage}\n  line: $line", e)
                  throw new PropagatedTribbleException(e.getMessage)
              }
            }
        }
    })
    VariantSampleMatrix(VariantMetadata(filters, sampleIds,
      Annotations.emptyIndexedSeq(sampleIds.length), Annotations.emptySignature,
      variantAnnotationSignatures), genotypes)
  }

}
