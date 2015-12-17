package org.broadinstitute.hail.methods

import org.broadinstitute.hail.vcf.BufferedLineIterator

import scala.io.Source
import org.apache.spark.{SparkConf, SparkContext}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.vcf
import org.broadinstitute.hail.annotations._
import scala.collection.convert._

object LoadVCF {
  // FIXME move to VariantDataset
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
        .takeWhile(line => line(0) == '#')
        .toArray
    }

    val codec = new htsjdk.variant.vcf.VCFCodec()
    val header = codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
      .asInstanceOf[htsjdk.variant.vcf.VCFHeader]

    val contigs = header.getContigLines.
    val contigRegex ="""##contig=<ID=(.+),length=(\d+)>""".r
    val contigLengths = {
      val contigMap = headerLines.map {
        case contigRegex(id, length) =>
          Some((id, length.toInt))
        case _ => None
      }.flatMap(i => i)
      .toMap

      if (contigMap.nonEmpty)
        contigMap
      else
        null
    }

    val annoRegex = """##INFO=<ID=(.+),Number=(.+),Type=(.+),Description="(.*)">""".r
    val annotationTypes = {
      val annotationMap = headerLines.flatMap {
        case annoRegex(id, number, typeOf, desc) => Some(id, VCFSignature.parse(number, typeOf, desc))
        case _ => None
      }.toMap

      if (annotationMap.nonEmpty)
        annotationMap
      else
        Map.empty[String, AnnotationSignature]
    }
    val annotationSignatures: AnnotationSignatures = Annotations[AnnotationSignature](Map("info" -> annotationTypes),
      Map("filters" -> new VCFSignature("Set[String]", "toSetString", "filters applied to site"),
        "pass" -> new VCFSignature("Boolean", "toBoolean", "filters were applied to vcf and this site passed"),
        "multiallelic" -> new VCFSignature("Boolean", "toBoolean", "Site is a split multiallelic"),
        "qual" -> new VCFSignature("Double", "toDouble", "vcf qual field"),
        "rsid" -> new VCFSignature("String", "toString", "site rdID")))

    val headerLine = headerLines.last
    assert(headerLine(0) == '#' && headerLine(1) != '#')

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val sampleAnnotations = Annotations.emptyOfArrayString(sampleIds.length)
    val sampleAnnotationSignatures = Annotations.emptyOfSignature()

    val headerLinesBc = sc.broadcast(headerLines)
    val genotypes = sc.textFile(file, nPartitions.getOrElse(sc.defaultMinPartitions))
      .mapPartitions { lines =>
        val reader = vcf.HtsjdkRecordReader(headerLinesBc.value)
        lines.filter(line => !line.isEmpty && line(0) != '#')
          .flatMap(reader.readRecord)
          .map { case (v, va, gs) =>
            val b = new GenotypeStreamBuilder(v, compress)
            for (g <- gs)
              b += g
            (v, va, b.result(): Iterable[Genotype])
          }
      }

    VariantSampleMatrix(VariantMetadata(contigLengths, sampleIds,
      headerLines, sampleAnnotations, sampleAnnotationSignatures, annotationSignatures), genotypes)
  }
}
