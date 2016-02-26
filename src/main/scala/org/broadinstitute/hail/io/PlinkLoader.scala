package org.broadinstitute.hail.io

import java.io._

import org.apache.hadoop.io.LongWritable
import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

case class SampleInfo(sampleIds: Array[String], pedigree: Pedigree)


class VariantParser(bimPath: String)
  extends Serializable {
  @transient var variants: Option[Array[Variant]] = None

  def parseBim(): Array[Variant] = {
    val bimFile = new File(bimPath)
    Source.fromFile(bimFile)
      .getLines()
      .filter(line => !line.isEmpty)
      .map { line =>
        val Array(contig, rsId, morganPos, bpPos, allele1, allele2) = line.split("\\s+")
        Variant(contig, bpPos.toInt, allele2, allele1)
      }
      .toArray
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
  private def parseFam(famPath: String): SampleInfo = {
    val famFile = new File(famPath)
    val sampleArray = Source.fromFile(famFile)
      .getLines()
      .filter(line => !line.isEmpty)
      .map { line => line.split("\\s+") }
      .toArray

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

    SampleInfo(sampleIds, ped)
  }

  private def parseBed(bedPath: String,
    sampleIds: Array[String],
    bimPath: String,
    sc: SparkContext): VariantDataset = {

    val nSamples = sampleIds.length

    sc.hadoopConfiguration.setInt("nSamples", nSamples)
    // FIXME what about withScope and assertNotStopped()?

    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[ParsedLine[Int]],
      sc.defaultMinPartitions)
    //rdd.cache()

    val variants = new VariantParser(bimPath).eval()

    println(s"rdd.count: ${rdd.count}")
    println(s"variant count: ${variants.size}")

    val variantRDD = rdd.map {
      case (lw, pl) => (variants(pl.getKey), Annotations.empty(), pl.getGS)
    }
    VariantSampleMatrix(VariantMetadata(sampleIds), variantRDD)
  }

  def apply(bfile: String, sc: SparkContext): VariantDataset = {
    apply(bfile + ".bed", bfile + ".bim", bfile + ".fam", sc)
  }

  def apply(bedFile: String, bimFile: String, famFile: String, sc: SparkContext): VariantDataset = {
    val samples = parseFam(famFile)
    println("nSamples is " + samples.sampleIds.length)

    val startTime = System.nanoTime()
    val vds = parseBed(bedFile, samples.sampleIds, bimFile, sc)
    val endTime = System.nanoTime()
    val diff = (endTime - startTime) / 1e9
    vds
  }
}