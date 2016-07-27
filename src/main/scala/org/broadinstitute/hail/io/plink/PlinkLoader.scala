package org.broadinstitute.hail.io.plink

import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.spark.SparkContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.utils.StringEscapeUtils._
import org.broadinstitute.hail.variant._

import scala.collection.mutable

case class SampleInfo(sampleIds: Array[String], annotations: IndexedSeq[Annotation], signatures: TStruct)

case class FamFileConfig(isQuantitative: Boolean, delimiter: String, missingValue: String)

object PlinkLoader {
  def expectedBedSize(nSamples: Int, nVariants: Long): Long = 3 + nVariants * ((nSamples + 3) / 4)

  private def parseBim(bimPath: String, hConf: Configuration): Array[Variant] = {
    readLines(bimPath, hConf)(_.map(_.transform { line =>
      line.value.split("\\s+") match {
        case Array(contig, rsId, morganPos, bpPos, allele1, allele2) => Variant(contig, bpPos.toInt, allele2, allele1)
        case other => fatal(s"Invalid .bim line.  Expected 6 fields, found ${other.length} ${plural(other.length, "field")}")
      }
    }
    ).toArray)
  }

  val numericRegex =
    """^-?(?:\d+|\d*\.\d+)(?:[eE]-?\d+)?$""".r

  def parseFam(filename: String, ffConfig: FamFileConfig,
    hConf: hadoop.conf.Configuration): (IndexedSeq[(String, Annotation)], Type) = {
    readLines(filename, hConf) { lines =>
      if (lines.isEmpty)
        fatal("Empty .fam file")

      val delimiter = unescapeString(ffConfig.delimiter)

      val phenoSig = if (ffConfig.isQuantitative) ("qPheno", TDouble) else ("isCase", TBoolean)

      val signature = TStruct(("famID", TString), ("patID", TString), ("matID", TString), ("isFemale", TBoolean), phenoSig)

      val kidSet = mutable.Set[String]()

      val m = lines.map {
        _.transform { line =>
          val split = line.value.split(delimiter)
          if (split.length != 6)
            fatal(s"Malformed .fam file: expected 6 fields, got ${split.length}")
          val Array(fam, kid, dad, mom, isFemale, pheno) = split

          if (kidSet(kid))
            fatal(s".fam sample name is not unique: $kid")
          else
            kidSet += kid

          val fam1 = if (fam != "0") fam else null
          val dad1 = if (dad != "0") dad else null
          val mom1 = if (mom != "0") mom else null
          val isFemale1 = isFemale match {
            case "0" => null
            case "1" => false
            case "2" => true
            case _ => fatal(s"Invalid sex: `$isFemale'. Male is `1', female is `2', unknown is `0'")
          }
          val pheno1 =
            if (ffConfig.isQuantitative)
              pheno match {
                case ffConfig.missingValue => null
                case numericRegex() => pheno.toDouble
                case _ => fatal(s"Invalid quantitative phenotype: `$pheno'. Value must be numeric or `${ffConfig.missingValue}'")
              }
            else
              pheno match {
                case ffConfig.missingValue => null
                case "1" => false
                case "2" => true
                case "0" => null
                case "-9" => null
                case numericRegex() => fatal(s"Invalid case-control phenotype: `$pheno'. Control is `1', case is `2', missing is `0', `-9', `${ffConfig.missingValue}', or non-numeric.")
                case _ => null
              }

          (kid, Annotation(fam1, dad1, mom1, isFemale1, pheno1))
        }
      }.toIndexedSeq
      (m, signature)
    }
  }

  private def parseBed(bedPath: String,
    sampleIds: IndexedSeq[String],
    sampleAnnotations: IndexedSeq[Annotation],
    sampleAnnotationSignature: Type,
    variants: Array[Variant],
    sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {

    val nSamples = sampleIds.length
    val variantsBc = sc.broadcast(variants)
    sc.hadoopConfiguration.setInt("nSamples", nSamples)

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[VariantRecord[Int]],
      nPartitions.getOrElse(sc.defaultMinPartitions))

    val variantRDD = rdd.map {
      case (lw, vr) => (variantsBc.value(vr.getKey), Annotation.empty, vr.getGS)
    }

    VariantSampleMatrix(VariantMetadata(
      sampleIds = sampleIds,
      sampleAnnotations = sampleAnnotations,
      globalAnnotation = Annotation.empty,
      saSignature = sampleAnnotationSignature,
      vaSignature = TStruct.empty,
      globalSignature = TStruct.empty,
      wasSplit = true), variantRDD)
  }

  def apply(bedPath: String, bimPath: String, famPath: String, ffConfig: FamFileConfig,
    sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {
    val (sampleInfo, signature) = parseFam(famPath, ffConfig, sc.hadoopConfiguration)
    val nSamples = sampleInfo.length
    if (nSamples <= 0)
      fatal(".fam file does not contain any samples")

    val variants = parseBim(bimPath, sc.hadoopConfiguration)
    val nVariants = variants.length
    if (nVariants <= 0)
      fatal(".bim file does not contain any variants")

    info(s"Found $nSamples samples in fam file.")
    info(s"Found $nVariants variants in bim file.")

    readFile(bedPath, sc.hadoopConfiguration) { dis =>
      val b1 = dis.read()
      val b2 = dis.read()
      val b3 = dis.read()

      if (b1 != 108 || b2 != 27)
        fatal("First two bytes of bed file do not match PLINK magic numbers 108 & 27")

      if (b3 == 0)
        fatal("Bed file is in individual major mode. First use plink with --make-bed to convert file to snp major mode before using Hail")
    }

    val bedSize = hadoopGetFileSize(bedPath, sc.hadoopConfiguration)
    if (bedSize != expectedBedSize(nSamples, nVariants))
      fatal("bed file size does not match expected number of bytes based on bed and fam files")

    if (bedSize < nPartitions.getOrElse(sc.defaultMinPartitions))
      fatal(s"The number of partitions requested (${nPartitions.getOrElse(sc.defaultMinPartitions)}) is greater than the file size ($bedSize)")

    val (ids, annotations) = sampleInfo.unzip

    val duplicateIds = ids.duplicates().toArray
    if (duplicateIds.nonEmpty) {
      val n = duplicateIds.length
      log.warn(s"found $n duplicate sample ${plural(n, "id")}:\n  ${duplicateIds.mkString("\n  ")}")
      warn(s"found $n duplicate sample ${plural(n, "id")}:\n  ${truncateArray(duplicateIds).mkString("\n  ")}")
    }

    val vds = parseBed(bedPath, ids, annotations, signature, variants, sc, nPartitions)
    vds
  }
}