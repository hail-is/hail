package org.broadinstitute.hail.io

import java.io._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.spark.{SparkContext}

import scala.io.Source

case class SampleInfo(sampleIds: Array[String], annotations: IndexedSeq[Annotations], signatures: Annotations)


class VariantParser(bimPath: String, hConf: Configuration)
  extends Serializable {
  @transient var variants: Option[Array[Variant]] = None

  def parseBim(): Array[Variant] = {
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


  def check() {
    require(variants.isEmpty)
    variants = Some(parseBim())
  }

  def eval(): Array[Variant] = variants match {
    case null | None =>
      val v = parseBim()
      println("number of variants is " + v.length)
      variants = Some(v)
      v
    case Some(v) => v
  }
}

object PlinkLoader {
  def expectedBedSize(nSamples:Int, nVariants:Long) : Long = 3 + nVariants*((nSamples / 4.0) + 0.75).toInt

  private def parseFam(famPath: String, hConf: Configuration): SampleInfo = {
    val sampleArray = readFile(famPath, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty)
        .map { line => line.split("\\s+") }
        .toArray
    }

    val signatures = Annotations(Map("fid" -> new SimpleSignature("String"),
      "mid" -> new SimpleSignature("String"),
      "pid" -> new SimpleSignature("String"),
      "reportedSex" -> new SimpleSignature("String"),
      "phenotype" -> new SimpleSignature("String")))

    val sampleIds = sampleArray.map { arr => arr(1) } //using IID as primary ID

    val sampleAnnotations = sampleArray.map{case Array(fid, iid, mat, pat, sex, pheno) =>
      Annotations(Map[String,Any]("fid" -> fid, "mid" -> mat, "pid" -> pat, "reportedSex" -> sex, "phenotype" -> pheno))
    }.toIndexedSeq

    val nSamples = sampleIds.length

    if (sampleIds.toSet.size != nSamples)
      fatal("Duplicate ids present in .fam file")

    println("number of samples is " + nSamples)

    SampleInfo(sampleIds, sampleAnnotations, signatures)
  }

  private def parseBed(bedPath: String,
    sampleInfo: SampleInfo,
    variants: Array[Variant],
    sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {

    val nSamples = sampleInfo.sampleIds.length

    sc.hadoopConfiguration.setInt("nSamples", nSamples)
    // FIXME what about withScope and assertNotStopped()?

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[ParsedLine[Int]],
      nPartitions.getOrElse(sc.defaultMinPartitions))

    val variantRDD = rdd.map {
      case (lw, pl) => (variants(pl.getKey), Annotations.empty(), pl.getGS)
    }

    VariantSampleMatrix(VariantMetadata(Array.empty[(String, String)],
      sampleInfo.sampleIds,
      sampleInfo.annotations,
      sampleInfo.signatures,
      Annotations.empty()), variantRDD)
  }

  def apply(bedPath: String, bimPath: String, famPath: String, sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {
    val sampleInfo = parseFam(famPath, sc.hadoopConfiguration)
    val nSamples = sampleInfo.sampleIds.length
    if (nSamples <= 0)
      fatal(".fam file does not contain any samples")

    val variants = new VariantParser(bimPath, sc.hadoopConfiguration).eval()
    val nVariants = variants.length
    if (nVariants <= 0)
      fatal(".bim file does not contain any variants")

    //check magic numbers in bed file
    readFile(bedPath, sc.hadoopConfiguration) {dis =>
      val b1 = dis.read()
      val b2 = dis.read()
      val b3 = dis.read()

      if (b1 != 108 || b2 != 27)
        fatal("First two bytes of bed file do not match PLINK magic numbers 108 & 27")

      if (b3 == 0)
        fatal("Bed file is in individual major mode. First use plink with --make-bed to convert file to snp major mode before using Hail")
    }

    // check num bytes matches fam and bim lengths
    val bedSize = hadoopGetFileSize(bedPath, sc.hadoopConfiguration)
    if (bedSize != expectedBedSize(nSamples, nVariants))
      fatal("bed file size does not match expected number of bytes based on bed and fam files")

    //check number of partitions requested is less than file size
    if (bedSize < nPartitions.getOrElse(sc.defaultMinPartitions))
      fatal(s"The number of partitions requested (${nPartitions.getOrElse(sc.defaultMinPartitions)}) is greater than the file size ($bedSize)")

    val startTime = System.nanoTime()
    val vds = parseBed(bedPath, sampleInfo, variants, sc, nPartitions)
    val endTime = System.nanoTime()
    val diff = (endTime - startTime) / 1e9
    vds
  }
}