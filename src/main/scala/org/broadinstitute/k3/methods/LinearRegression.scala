package org.broadinstitute.k3.methods

import scala.collection.mutable.ListBuffer
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
    require(ped.phenoDefinedForAll)

    val nPhenotypedSamples = cov.data.rows
    val rowOfSample = cov.sampleOfRow.zipWithIndex.toMap

    val y = DenseVector.zeros[Double](nPhenotypedSamples)
    for (i <- 0 until nPhenotypedSamples)
     y(i) = ped.phenoOf(cov.sampleOfRow(i)).toString.toDouble

    val allOnes = DenseMatrix.ones[Double](nPhenotypedSamples, 1)
    val q = qr.reduced.justQ(DenseMatrix.horzcat(allOnes, cov.data))
    val yp = y - q * (q.t * y)
    // FIXME: divide yp by sample standard dev, 1 / (n - 1)

    val sc = vds.sparkContext
    val nPhenotypedSamplesBc = sc.broadcast(nPhenotypedSamples)
    val rowOfSampleBc = sc.broadcast(rowOfSample)
    val isPhenotypedBc = sc.broadcast((0 until vds.nSamples).map(rowOfSample.isDefinedAt).toArray)
    val qBc = sc.broadcast(q)
    val ypBc = sc.broadcast(yp)

    new LinearRegression(vds
      .filterSamples(s => isPhenotypedBc.value(s))
      .mapWithKeys{ (v, s, g) => (v, (rowOfSampleBc.value(s), g.call.map(_.gt))) }
      .groupByKey()
      .mapValues{
        gts => {
          val n = nPhenotypedSamplesBc.value
          val x = SparseVector.zeros[Double](n)

          // replace missing values with mean
          val missingRowsBuffer = new ListBuffer[Int]()
          var gtSum = 0
          gts.foreach{
            case (row, gt) =>
              if (gt.isDefined) {
                if (gt.get != 0) {
                  gtSum += gt.get
                  x(row) = gt.get.toDouble
                }
              }
              else
                missingRowsBuffer += row
          }
          val missingRows = missingRowsBuffer.toList
          val mu = gtSum.toDouble / (n - missingRows.length)
          missingRows.foreach(row => x(row) = mu)

          // FIXME: standardize x, 1 / n

          val xp = x - qBc.value * (qBc.value.t * x)
          (xp dot ypBc.value) / (xp dot xp)  //may be able to take further advantage of sparsity in x
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
