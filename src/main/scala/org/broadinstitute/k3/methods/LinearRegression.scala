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
    val d = n - k - 2
    require(d > 0)

    // extract the phenotype vector y
    val rowOfSample = cov.sampleOfRow.zipWithIndex.toMap
    val y = DenseVector.zeros[Double](n)
    for (row <- 0 until n)
     y(row) = ped.phenoOf(cov.sampleOfRow(row)).toString.toDouble

    // augment covariate matrix with 1s vector, compute q.t and yyp = yp dot yp
    val qt = qr.reduced.justQ(DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), cov.data)).t
    val qty = qt * y
    val yyp = (y dot y) - (qty dot qty)

    val sc = vds.sparkContext
    val rowOfSampleBc = sc.broadcast(rowOfSample)
    val isPhenotypedBc = sc.broadcast((0 until vds.nSamples).map(rowOfSample.isDefinedAt).toArray)

    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast(yyp)

    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    new LinearRegression(vds
      .filterSamples(s => isPhenotypedBc.value(s))
      .mapWithKeys{ (v, s, g) => (v, (rowOfSampleBc.value(s), g.call.map(_.gt))) }
      .groupByKey()
      .mapValues{
        gts => {
          val qt = qtBc.value
          val n = qt.cols
          val d = n - qt.rows - 1 // qt includes vector of 1s

          // replace missing values with mean
          val y = yBc.value
          val x = SparseVector.zeros[Double](n)
          val missingRowsBuffer = new ListBuffer[Int]()
          var sumX = 0
          var sumXX = 0
          var sumXY = 0.0
          gts.foreach{
            case (row, gt) =>
              if (gt.isDefined) {
                if (gt.get != 0) {
                  sumX += gt.get
                  sumXX += gt.get * gt.get
                  sumXY += gt.get * y(row)
                  x(row) = gt.get.toDouble
                }
              }
              else
                missingRowsBuffer += row
          }
          assert(sumX > 0) //FIXME: better error handling
          val missingRows = missingRowsBuffer.toList
          val nMissing = missingRows.length
          val mu = sumX.toDouble / (n - nMissing)
          missingRows.foreach(row => x(row) = mu)

          // these are lines useful for normalization, unnecessary for regression
          // val sum = gtSum + nMissing * mu
          // val sumSqCentered = math.sqrt(sumSq - sum * sum / n) // = (x - mu) dot (x - mu)

          // compute regression coef and stats for 2-sided t test with d degrees of freedom
          val xx = sumXX + mu * mu * nMissing
          val xy = sumXY + mu * missingRows.map(row => y(row)).sum

          val qtx = qt * x
          val qty = qtyBc.value

          val xxp = xx - (qtx dot qtx)
          val xyp = xy - (qtx dot qty)
          val yyp = yypBc.value

          val b = xyp / xxp
          val se = math.sqrt((yyp / xxp - b * b) / d)
          val t = b / se
          val p = 2 * tDistBc.value.cumulativeProbability(- math.abs(t))

          LinRegOutput(nMissing, b, se, t, p)
        }
      }
    )
  }
}

case class LinearRegression(lr: RDD[(Variant, LinRegOutput)]) {

  def write(filename: String) {
    def toLine(v: Variant, lro: LinRegOutput) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt +
      "\t" + lro.nMissing + "\t" + lro.beta + "\t" + lro.stdError + "\t" + lro.t + "\t" + lro.p
    lr.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tMISS\tBETA\tSE\tT\tP\n")
  }
}
