package org.broadinstitute.hail.io

import java.io._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.spark.{SparkContext}

import scala.io.Source

case class SampleInfo(sampleIds: Array[String], pedigree: Pedigree)


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
  private def parseFam(famPath: String, hConf: Configuration): SampleInfo = {
    val sampleArray = readFile(famPath, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty)
        .map { line => line.split("\\s+") }
        .toArray
    }

    val sampleIds = sampleArray.map { arr => arr(1) }

    val nSamples = sampleIds.length
    val indexOfSample: Map[String, Int] = sampleIds.zipWithIndex.toMap
    def maybeId(id: String): Option[Int] = if (id != "0") indexOfSample.get(id) else None
    def stringOrZero(str: String): Option[String] = if (str != "0") Some(str) else None

    val ped = Pedigree(sampleArray.map { arr =>
      val Array(fam, kid, dad, mom, sex, pheno) = arr
      Trio(indexOfSample(kid), stringOrZero(fam), maybeId(dad), maybeId(mom),
        Sex.withNameOption(sex), Phenotype.withNameOption(pheno))
    })

    println("number of samples is " + nSamples)
    SampleInfo(sampleIds, ped)
  }

  private def parseBed(bedPath: String,
    sampleIds: Array[String],
    variants: Array[Variant],
    sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {

    val nSamples = sampleIds.length

    sc.hadoopConfiguration.setInt("nSamples", nSamples)
    // FIXME what about withScope and assertNotStopped()?

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[ParsedLine[Int]],
      nPartitions.getOrElse(sc.defaultMinPartitions))

    val variantRDD = rdd.map {
      case (lw, pl) => (variants(pl.getKey), Annotations.empty(), pl.getGS)
    }

    VariantSampleMatrix(VariantMetadata(sampleIds), variantRDD)
  }

  def apply(bedPath: String, bimPath: String, famPath: String, sc: SparkContext, nPartitions: Option[Int] = None): VariantDataset = {
    val samples = parseFam(famPath, sc.hadoopConfiguration)
    val nSamples = samples.sampleIds.length
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
    if (bedSize != (3 + nVariants * ((nSamples / 4.00) + .75).toInt))
      fatal("bed file size does not match expected number of bytes based on bed and fam files")

    //check number of partitions requested is less than file size
    if (bedSize < nPartitions.getOrElse(sc.defaultMinPartitions))
      fatal(s"The number of partitions requested (${nPartitions.getOrElse(sc.defaultMinPartitions)}) is greater than the file size ($bedSize)")

    val startTime = System.nanoTime()
    val vds = parseBed(bedPath, samples.sampleIds, variants, sc, nPartitions)
    val endTime = System.nanoTime()
    val diff = (endTime - startTime) / 1e9
    vds
  }
}