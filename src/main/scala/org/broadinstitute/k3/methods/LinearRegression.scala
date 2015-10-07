package org.broadinstitute.k3.methods

import scala.io.Source
import java.io.File
import breeze.linalg._
import breeze.stats.{mean, meanAndVariance}
import org.apache.spark.rdd.RDD

import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._

object CovariateData {

  def read(filename: String, sampleIds: Array[String]): CovariateData = {
    val src = Source.fromFile(new File(filename))
    val (header, lines) = src.getLines().filter(line => !line.isEmpty).toList.splitAt(1)
    src.close()

    val covOfCol = header.head.split("\\s+").tail
    val nCov = covOfCol.length
    val nSamples = lines.length

    val sampleOfRow = Array.ofDim[Int](nSamples)
    val data = DenseMatrix.zeros[Double](nSamples, nCov)

    val indexOfSample: Map[String, Int] = sampleIds.zipWithIndex.toMap

    var row = Array.ofDim[String](nCov)
    for (i <- 0 until nSamples) {
      val line = lines(i).split("\\s+")
      sampleOfRow(i) = indexOfSample(line(0))
      for (j <- 0 until nCov) {
        data(i, j) = line(j+1).toDouble
      }
    }
    CovariateData(sampleOfRow, covOfCol, data)
  }
}

case class CovariateData(sampleOfRow: Array[Int], covariateOfCol: Array[String], data: DenseMatrix[Double])

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    require(vds.sampleIds.length == cov.data.rows)

    val nSamples = cov.data.rows
    val y = DenseVector.zeros[Double](nSamples)

    for (i <- 0 until nSamples)
     y(i) = ped.phenoOf(cov.sampleOfRow(i)).toString.toDouble

    val allOnes = DenseMatrix.ones[Double](nSamples, 1)

    val q = qr.reduced.justQ(DenseMatrix.horzcat(allOnes, cov.data))

    val yp = y - q * (q.t * y)

    val sc = vds.sparkContext
    val qBc = sc.broadcast(q)
    val ypBc = sc.broadcast(yp)
    val nSamplesBc = sc.broadcast(nSamples)

    new LinearRegression(vds
      .mapWithKeys{ (v, s, g) => (v, g.call.get.gt) }
      .groupByKey()
      .mapValues{
        gs => {
          val x = new DenseMatrix(nSamplesBc.value, 1, gs.map(_.toDouble).toArray)
          val q = qBc.value
          val xp: DenseMatrix[Double] = x - q * (q.t * x)
          val r = xp \ (ypBc.value: DenseVector[Double])
          r(0)
        }
      }
    )
  }
}

case class LinearRegression(betas: RDD[(Variant, Double)]) {

  def write(filename: String) {
    def toLine(v: Variant, beta: Double) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\t" + beta
    betas.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tBETA\n")
  }
}
