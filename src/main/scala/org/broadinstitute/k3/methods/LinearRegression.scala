package org.broadinstitute.k3.methods

import scala.io.Source
import java.io.File
import breeze.linalg._
import org.apache.spark.rdd.RDD

import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._

object CovariateData {

  def read(filename: String, sampleIds: Array[String]): CovariateData = {
    val src = Source.fromFile(new File(filename))
    val (header, lines) = src.getLines().filter(line => !line.isEmpty).toList.splitAt(1)
    src.close()

    val covIds = header.head.split("\\s+").tail
    val nCov = covIds.length
    val nSamples = lines.length

    val rowIds = Array.ofDim[Int](nSamples)
    val data = DenseMatrix.zeros[Double](nSamples, nCov)

    val indexOfSample: Map[String, Int] = sampleIds.zipWithIndex.toMap

    var row = Array.ofDim[String](nCov)
    for (i <- 0 until nSamples) {
      val line = lines(i).split("\\s+")
      rowIds(i) = indexOfSample(line(0))
      for (j <- 0 until nCov) {
        data(i, j) = line(j+1).toDouble
      }
    }
    CovariateData(rowIds, covIds, data)
  }
}

case class CovariateData(rowIds: Array[Int], covIds: Array[String], data: DenseMatrix[Double])

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    require(vds.sampleIds.length == cov.data.rows)

    LinearRegression(vds.variants.map((_, 0.0)))
  }
}

case class LinearRegression(betas: RDD[(Variant, Double)]) {

  def write(filename: String) {
    def toLine(v: Variant, beta: Double) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\t" + beta
    betas.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tBETA\n")
  }
}
