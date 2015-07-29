package org.broadinstitute.k3.methods

import java.io.FileInputStream
import java.util.zip.GZIPInputStream

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.rdd._

import org.broadinstitute.k3.variant._

/**
 * Created by cseed on 7/24/15.
 */
object LoadVCF {
  def parseGenotype(entry: String) = {
    val genotypeRegex =
      """([\d\.])[/|]([\d\.]):(\d+,\d+|\.):(\d+|\.):(\d+|\.):(\d+,\d+,\d+|\.)""".r
    val genotypeRegex(gtStr1, gtStr2, adStr, dpStr, gqStr, plStr) =
      entry

    val gt =
      if (gtStr1 == ".")
        -1
      else
        gtStr1.toInt + gtStr2.toInt

    val ad =
      if (adStr == ".")
        (-1, -1)
      else {
        val adList = adStr.split(",").map(_.toInt)
        (adList(0), adList(1))
      }

    val dp =
      if (dpStr == ".")
        -1
      else
        dpStr.toInt

    val (pl1, pl2) =
      if (plStr == ".")
        (-1, -1)
      else {
        val plList = plStr.split(",").map(_.toInt)
        gt match {
          case 0 => (plList(1), plList(2))
          case 1 => (plList(0), plList(2))
          case 2 => (plList(0), plList(1))
        }
      }

    Genotype(gt, ad, dp, pl1, pl2, null)
  }

  def apply(sc: SparkContext, file: String): RDD[((String, Variant), Genotype)] = {
    // FIXME move to util
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

    def parseLine(line: String): Array[((String, Variant), Genotype)] = {
      val words: Array[String] = line.split("\t")
      val variant = Variant(words(0),
        words(1).toInt,
        DNASeq(words(3)),
        DNASeq(words(4)))
      words.drop(9)
        .zip(sampleIds)
        .map({ case (entry, sample) => ((sample, variant), parseGenotype(entry)) })
    }

    val loadFile =
      if (file.takeRight(5) == ".vcfd")
        file + "/shard*.gz"
      else
        file

    val linesRDD = sc.textFile(loadFile)
    linesRDD
      .filter(_(0) != '#')
      .flatMap(parseLine)
  }
}
