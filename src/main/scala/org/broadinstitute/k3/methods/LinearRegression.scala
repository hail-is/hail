package org.broadinstitute.k3.methods

import scala.collection.mutable.ListBuffer
import scala.io.Source
import java.io.File
import breeze.linalg._
import breeze.stats.meanAndVariance
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
    require(ped.phenoDefinedForAll)

    val n = cov.data.rows
    val k = cov.data.cols
    require(n - k - 2 > 0)

    val rowOfSample = cov.sampleOfRow.zipWithIndex.toMap

    // extract the phenotype vector y
    val y = DenseVector.zeros[Double](n)
    for (i <- 0 until n)
     y(i) = ped.phenoOf(cov.sampleOfRow(i)).toString.toDouble

    // augment the covariate matrix and compute q
    val q = qr.reduced.justQ(DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), cov.data))

    //project y to yp, automatically mean centered, then normalize yp
    val yp = y - q * (q.t * y)
    val ypYp = yp dot yp

    val sc = vds.sparkContext
    val rowOfSampleBc = sc.broadcast(rowOfSample)
    val isPhenotypedBc = sc.broadcast((0 until vds.nSamples).map(rowOfSample.isDefinedAt).toArray)
    val qBc = sc.broadcast(q)
    val ypBc = sc.broadcast(yp)
    val ypYpBc = sc.broadcast(ypYp)
//    val ypNormBc = sc.broadcast(ypNorm)

    new LinearRegression(vds
      .filterSamples(s => isPhenotypedBc.value(s))
      .mapWithKeys{ (v, s, g) => (v, (rowOfSampleBc.value(s), g.call.map(_.gt))) }
      .groupByKey()
      .mapValues{
        gts => {
          val n = qBc.value.rows
          val k = qBc.value.cols
          val x = SparseVector.zeros[Double](n)

          // replace missing values with mean
          val missingRowsBuffer = new ListBuffer[Int]()
          var gtSum = 0
          var gtSumSq = 0
          gts.foreach{
            case (row, gt) =>
              if (gt.isDefined) {
                if (gt.get != 0) {
                  gtSum += gt.get
                  gtSumSq += gt.get * gt.get
                  x(row) = gt.get.toDouble
                }
              }
              else
                missingRowsBuffer += row
          }
          assert(gtSum > 0) //FIXME: better error handling

          // fill in missing gt with mean
          val missingRows = missingRowsBuffer.toList
          val nMissing = missingRows.length
          val mu = gtSum.toDouble / (n - nMissing)
          missingRows.foreach(row => x(row) = mu)

          val sum = gtSum + nMissing * mu // could stick to integer arithmetic longer
          val sumSq = gtSumSq + nMissing * mu * mu
          val normOfCenteredX = math.sqrt(sumSq - sum * sum / n)
          x :-= mu
          x :/= normOfCenteredX  // FIXME: push sparse vectors down to computation of xx, xy

          val q = qBc.value
          val yp = ypBc.value
          val xp = x - q * (q.t * x)

          val xx = xp dot xp
          val xy = xp dot yp
          val yy = ypYpBc.value
          
          val d = n - k - 2
          val t = math.sqrt(d * xy * xy / (xx * yy - xy * xy)) //direct formula, d degrees of freedom
          val b = xy / xx
          val se = b / t // = math.sqrt((yy - 2 * b * xy + b * b * xx) / (d * xx))
          t

          // FIXME: need to normalize statistic based on y as well
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
