package org.broadinstitute.k3.methods

import java.io.FileInputStream
import java.util.zip.GZIPInputStream

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.rdd._

import org.broadinstitute.k3.variant._

object LoadVCF {
  def parseGenotype(entry: String) = {
    val genotypeRegex =
      """([\d\.])[/|]([\d\.]):(\d+,\d+|\.):(\d+|\.):(\d+|\.):(\d+,\d+,\d+|\.)""".r
    val genotypeRegex(gtStr1, gtStr2, adStr, dpStr, gqStr, plStr) = entry

    val gt =
      if (gtStr1 == ".")
        -1
      else
        gtStr1.toInt + gtStr2.toInt

    val ad =
      if (adStr == ".")
        (0, 0)
      else {
        val adList = adStr.split(",").map(_.toInt)
        assert(adList.length == 2)
        (adList(0), adList(1))
      }

    val dp =
      if (dpStr == ".")
        0
      else
        dpStr.toInt

    val pl =
      if (plStr == ".")
        null
      else {
        val plList = plStr.split(",").map(_.toInt)
        val minPl = plList.min
        assert(plList.length == 3)
        (plList(0) - minPl, plList(1) - minPl, plList(2) - minPl)
      }

    val normalizedPl =
      gt match {
        case -1 => null
        case 0 => (0, pl._2, pl._3)
        case 1 => (pl._1, 0, pl._3)
        case 2 => (pl._1, pl._2, 0)
      }

    Genotype(gt, ad,
      dp max (ad._1 + ad._2),
      normalizedPl)
  }

  def apply(sc: SparkContext, file: String): VariantDataset = {
    val s = if (file.takeRight(7) == ".vcf.gz")
      Source.fromInputStream(new GZIPInputStream(new FileInputStream(file)))
    else if (file.takeRight(5) == ".vcfd")
      Source.fromFile(file + "/header")
    else {
      assert(file.takeRight(4) == ".vcf")
      Source.fromFile(file)
    }

    val headerLine = s
                     .getLines()
                     .find(line => line(0) == '#' && line(1) != '#')
                     .get
    val sampleIds = headerLine
                    .split("\t")
                    .drop(9)

    def parseLine(line: String): (Variant, GenotypeStream) = {
      val words: Array[String] = line.split("\t")
      val variant = Variant(words(0),
        words(1).toInt,
        words(3),
        words(4))

      // FIXME foreach can't be right
      val b = new GenotypeStreamBuilder(variant)
      words.drop(9)
      .map(parseGenotype)
      .foreach(g => b.+=((0, g)))
      val a = b.result()

      // println("uncompLen = " + a.length)

      (variant, b.result())
    }

    val loadFile =
      if (file.takeRight(5) == ".vcfd")
        file + "/shard*.gz"
      else
        file

    val linesRDD = sc.textFile(loadFile)
    val variantRDD = linesRDD
                     .filter(_(0) != '#')
                     .map(parseLine)

    new VariantDataset(sampleIds, variantRDD)
  }
}
