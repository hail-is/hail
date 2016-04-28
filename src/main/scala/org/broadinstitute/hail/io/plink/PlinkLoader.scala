package org.broadinstitute.hail.io.plink

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.spark.SparkContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.io._

import scala.io.Source

case class SampleInfo(sampleIds: Array[String], annotations: IndexedSeq[Annotation], signatures: TStruct)

object PlinkLoader {
  def expectedBedSize(nSamples:Int, nVariants:Long) : Long = 3 + nVariants * ((nSamples + 3) / 4)

  private def parseBim(bimPath: String, hConf: Configuration): Array[Variant] = {
    readFile(bimPath, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty)
        .map { line =>
          val Array(contig, rsId, morganPos, bpPos, allele1, allele2) = line.split("\\s+")
          Variant(contig, bpPos.toInt, allele2, allele1)
        }
        .toArray
    }
  }

  private def parseFam(famPath: String, hConf: Configuration): SampleInfo = {
    val sampleArray = readFile(famPath, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty)
        .map { line => line.split("\\s+") }
        .toArray
    }

    val signatures = TStruct("fid" -> TString, "mid" -> TString, "pid" -> TString, "reportedSex" -> TString, "phenotype" -> TString)

    val sampleIds = sampleArray.map { arr => arr(1) } //using IID as primary ID

    val sampleAnnotations = sampleArray.map{case Array(fid, iid, mid, pid, sex, pheno) =>
      Annotation(fid, mid, pid, sex, pheno)
    }.toIndexedSeq

    val nSamples = sampleIds.length

    if (sampleIds.toSet.size != nSamples)
      fatal("Duplicate ids present in .fam file")

    SampleInfo(sampleIds, sampleAnnotations, signatures)
  }

  private def parseBed(bedPath: String,
    sampleInfo: SampleInfo,
    variants: Array[Variant],
    sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {

    val nSamples = sampleInfo.sampleIds.length
    val variantsBc = sc.broadcast(variants)
    sc.hadoopConfiguration.setInt("nSamples", nSamples)

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[VariantRecord[Int]],
      nPartitions.getOrElse(sc.defaultMinPartitions))

    val variantRDD = rdd.map {
      case (lw, pl) => (variantsBc.value(pl.getKey), Annotation.empty, pl.getGS)
    }

    VariantSampleMatrix(VariantMetadata(
      sampleInfo.sampleIds,
      sampleInfo.annotations,
      Annotation.empty,
      sampleInfo.signatures,
      TStruct.empty,
      TStruct.empty), variantRDD)
  }

  def apply(bedPath: String, bimPath: String, famPath: String, sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {
    val sampleInfo = parseFam(famPath, sc.hadoopConfiguration)
    val nSamples = sampleInfo.sampleIds.length
    if (nSamples <= 0)
      fatal(".fam file does not contain any samples")

    val variants = parseBim(bimPath, sc.hadoopConfiguration)
    val nVariants = variants.length
    if (nVariants <= 0)
      fatal(".bim file does not contain any variants")

    info(s"Found $nSamples samples in fam file.")
    info(s"Found $nVariants variants in bim file.")

    readFile(bedPath, sc.hadoopConfiguration) {dis =>
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

    val vds = parseBed(bedPath, sampleInfo, variants, sc, nPartitions)
    vds
  }
}