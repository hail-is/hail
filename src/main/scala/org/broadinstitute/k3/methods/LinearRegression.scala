package org.broadinstitute.k3.methods

import scala.collection.mutable.ListBuffer
import scala.io.Source
import java.io.File
import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.apache.commons.math3.distribution.TDistribution

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

case class LinRegOutput(nMissing: Int, beta: Double, stdError: Double, t: Double, p: Double)

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    require(ped.phenoDefinedForAll)

    val n = cov.data.rows
    val k = cov.data.cols
    require(n - k - 2 > 0)

    // extract the phenotype vector y
    val rowOfSample = cov.sampleOfRow.zipWithIndex.toMap
    val y = DenseVector.zeros[Double](n)
    for (i <- 0 until n)
     y(i) = ped.phenoOf(cov.sampleOfRow(i)).toString.toDouble

    // augment the covariate matrix with ones vector, and compute q.t, and yy = yp dot yp
    val qt = qr.reduced.justQ(DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), cov.data)).t
    val qty = qt * y
    val yy = (y dot y) - (qty dot qty) // = yp dot yp for yp = (I - q * q.t) y

    val sc = vds.sparkContext
    val rowOfSampleBc = sc.broadcast(rowOfSample)
    val isPhenotypedBc = sc.broadcast((0 until vds.nSamples).map(rowOfSample.isDefinedAt).toArray)

    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yyBc = sc.broadcast(yy)

    new LinearRegression(vds
      .filterSamples(s => isPhenotypedBc.value(s))
      .mapWithKeys{ (v, s, g) => (v, (rowOfSampleBc.value(s), g.call.map(_.gt))) }
      .groupByKey()
      .mapValues{
        gts => {
          val qt = qtBc.value
          val n = qt.cols
          val k = qt.rows

          // replace missing values with mean
          val x = SparseVector.zeros[Double](n)
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
          val missingRows = missingRowsBuffer.toList
          val nMissing = missingRows.length
          val mu = gtSum.toDouble / (n - nMissing)
          missingRows.foreach(row => x(row) = mu)

          // divide x by norm of centered x
          val sum = gtSum + nMissing * mu
          val sumSq = gtSumSq + nMissing * mu * mu
          x :/= math.sqrt(sumSq - sum * sum / n)

          // calculate dot products between xp and yp
          val qtx = qt * x
          val qty = qtyBc.value
          val xx = (x dot x) - (qtx dot qtx) // = xp dot xp for xp = (I - q * q.t) x
          val xy = (x dot y) - (qtx dot qty) // = xp dot yp
          val yy = yyBc.value                // = yp dot yp

          // compute regression coef and stats for 2-sided t test with d degrees of freedom
          val signT = if (xy > 0) 1 else -1
          val d = n - k - 2
          val b = xy / xx
          val t = signT * math.sqrt(d * xy * xy / (xx * yy - xy * xy))
          val se = b / t // = sqrt( 1/d * |y - bx|^2 / xx )
          val tDist = new TDistribution(null, d.toDouble)
          val p = 2 * tDist.cumulativeProbability(-1 * signT * t)

          LinRegOutput(nMissing, b, se, t, p)
        }
      }
    )
  }
}

case class LinearRegression(lr: RDD[(Variant, LinRegOutput)]) {

  def write(filename: String) {
    def toLine(v: Variant, lro: LinRegOutput) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt +
      "\t" + lro.nMissing + "\t" + lro.beta + lro.stdError + "\t" + lro.t + "\t" + lro.p
    lr.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tMISS\tBETA\tSE\tT\tP\n")
  }
}
