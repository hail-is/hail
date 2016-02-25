package org.broadinstitute.hail.io

import java.io._

import org.apache.hadoop.io.LongWritable
import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source
import scala.reflect.ClassTag

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
        Variant(contig, bpPos.toInt, allele1, allele2)
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

    def plinkToHail(call: Int): Int = {
      if (call == 0)
        call
      else if (call == 1)
        -1
      else
        call - 1
    }

    sc.hadoopConfiguration.setInt("nSamples", nSamples)
    // FIXME what about withScope and assertNotStopped()?
    val rdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[LongWritable], classOf[ParsedLine[Int]],
      sc.defaultMinPartitions)

    val variants = new VariantParser(bimPath).eval()
    val indices = rdd.map { case (lw, pl) => pl.getKey}
      .collect()
    println(s"first 5: ${indices.take(5).mkString(",")}")
    println(s"last 5: ${indices.takeRight(5).mkString(",")}")

    val variantRDD = rdd.map {
      case (lw, pl) => (variants(pl.getKey), pl.getGS)
    }
    VariantSampleMatrix(VariantMetadata(null, sampleIds), variantRDD)
  }


  def apply(fileBase: String, sc: SparkContext): VariantDataset = {
    val samples = parseFam(fileBase + ".fam")
    println("nSamples is " + samples.sampleIds.length)

    val startTime = System.nanoTime()
    val vds = parseBed(fileBase + ".bed", samples.sampleIds, fileBase + ".bim", sc)
    val endTime = System.nanoTime()
    val diff = (endTime - startTime) / 1e9
    vds
  }
}