package org.broadinstitute.hail.methods

import scala.io.Source
import org.apache.spark.{SparkConf, SparkContext}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.vcf
import org.broadinstitute.hail.annotations._

object LoadVCF {
  // FIXME move to VariantDataset

  val arrayRegex = """Array\[(\w+)\]""".r
  val setRegex = """Set\[(\w+)\]""".r
  def getConversionMethod(str: String): String = {
    str match {
      case arrayRegex(subType) => s"toArray$subType"
      case setRegex(subType) => s"toSet$subType"
      case _ => s"to$str"
    }
  }

  def parseInfoType(str: String): String = {
    str match {
      case "Flag" => "Boolean"
      case "Integer" => "Int"
      case "Float" => "Double"
      case "String" => "String"
      case "Character" => "String"
      case _ => throw new UnsupportedOperationException("unexpected annotation type")
    }
  }

  def parseInfoLine(number: String, typeOf: String, desc: String): AnnotationSignature = {
    val parsedType = parseInfoType(typeOf)
    if (number == "0" || number == "1") {
      new SimpleSignature(parsedType, getConversionMethod(parsedType), desc)
    }
    else if (number == "A" || number == "R" || number == "G") {
      val arrType = s"Array[$parsedType]"
      new SimpleSignature(arrType, getConversionMethod(arrType), desc)
    }
    else if (number == "." && parsedType == "String") {
      new SimpleSignature(parsedType, getConversionMethod(parsedType), desc)
    }
    else
      throw new UnsupportedOperationException
  }

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
//    contigLengths.foreach { case (id, len) => println("contig=%s, length=%s".format(id, len))}

    val annoRegex = """##INFO=<ID=(.+),Number=(.+),Type=(.+),Description="(.*)">""".r
    val annotationTypes = {
      val annotationMap = headerLines.map {
        case annoRegex(id, number, typeOf, desc) => Some(id, parseInfoLine(number, typeOf, desc))
        case _ => None
      }.flatMap(i => i)
        .toMap

      if (annotationMap.nonEmpty)
        annotationMap
      else
        Map.empty[String, AnnotationSignature]
    }
    val annotationSignatures: AnnotationSignatures = Annotations[AnnotationSignature](Map("info" -> annotationTypes),
      Map("filters" -> new SimpleSignature("Set[String]", "toSetString", "filters applied to site"),
        "pass" -> new SimpleSignature("Boolean", "toBoolean", "filters were applied && this site passed"),
        "multiallelic" -> new SimpleSignature("Boolean", "toBoolean", "Site is a split multiallelic"),
        "qual" -> new SimpleSignature("Double", "toDouble", "vcf qual field"),
        "rsid" -> new SimpleSignature("String", "toString", "site rdID")))
//    annotationTypes.foreach {
//      case (id, v) => println(s"id=$id, type=$v")
//    }

    val headerLine = headerLines.last
    assert(headerLine(0) == '#' && headerLine(1) != '#')

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val sampleAnnotations = EmptySampleAnnotations(sampleIds.length)
    val sampleAnnotationSignatures = EmptyAnnotationSignatures()

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
