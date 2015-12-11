package org.broadinstitute.hail.io

import java.io._

import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

case class SampleInfo(sampleIds: Array[String], pedigree: Pedigree)

object PlinkLoader {
  val sparseGt = Array(Genotype(-1, (0, 0), 0, (0,0,0)),
    Genotype(0, (0, 0), 0, (0,0,0)),
    Genotype(1, (0, 0), 0, (0,0,0)),
    Genotype(2, (0, 0), 0, (0,0,0)))

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

  private def parseBim(bimPath: String): Array[Variant] = {
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

  private def parseBed(bedPath: String,
    sampleIds: Array[String],
    variants: Array[Variant],
    sc: SparkContext,
    vsmType: String): VariantDataset = {

    val nSamples = sampleIds.length
    val nVariants = variants.length

    // 2 bits per genotype, round up to the nearest byte.  Add 3 for magic numbers at the start of the file
    def getStart(variantIndex: Int): Long = 3 + variantIndex.toLong * ((sampleIds.length / 4.00) + .75).toInt

    def plinkToHail(call: Int): Int = {
      if (call == 0)
        call
      else if (call == 1)
        -1
      else
        call - 1
    }

    val blocks = variants.zipWithIndex.map { case (v, i) => (v, getStart(i)) }

      sc.hadoopConfiguration.setInt("nSamples", nSamples)
    // FIXME what about withScope and assertNotStopped()?
    val bbRdd = sc.hadoopFile(bedPath, classOf[PlinkInputFormat], classOf[RichLongWritable], classOf[ByteBlock],
      sc.defaultMinPartitions)

    val variantsBc = sc.broadcast(variants)
    def parseVariant(bb: ByteBlock):
    (Variant, Iterable[Genotype]) = {
      val variant = variantsBc.value(bb.getIndex)
      val bar = new ByteArrayReader(bb.getArray)
      val b = new GenotypeStreamBuilder(variant, compress = false)
      bar.readBytes(bar.length)
        .iterator
        .flatMap { i => Iterator(i & 3, (i >> 2) & 3, (i >> 4) & 3, (i >> 6) & 3) }
        .map(plinkToHail)
        .take(nSamples)
        .foreach(i => b += sparseGt(i+1))
      (variant, b.result(): Iterable[Genotype])
    }

    val vgsRdd = bbRdd.map {
      case (l, bb) => parseVariant(bb)
    }
    VariantSampleMatrix(VariantMetadata(null, sampleIds), vgsRdd)
  }


  def apply(fileBase: String, sc: SparkContext, vsmType: String = "sparky"): VariantDataset = {
    val samples = parseFam(fileBase + ".fam")
    println("nSamples is " + samples.sampleIds.length)

    val variants = parseBim(fileBase + ".bim")
    println("nVariants is " + variants.length)

    val startTime = System.nanoTime()
    val vds = parseBed(fileBase + ".bed", samples.sampleIds, variants, sc, vsmType)
    val endTime = System.nanoTime()
    val diff = (endTime - startTime) / 1e9
    vds
  }
}