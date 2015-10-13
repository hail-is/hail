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

case class LinRegOutput(nMissing: Int, beta: Double, stdError: Double, t: Double)

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

    // augment the covariate matrix with ones vector, and compute q.t, project y
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

          val sum = gtSum + nMissing * mu
          val sumSq = gtSumSq + nMissing * mu * mu
          println(x)
          x :/= math.sqrt(sumSq - sum * sum / n)

          val qtx = qt * x
          val qty = qtyBc.value

          // these turn out to be unnecessary!
          // val xp = x - qt.t * (qt * x)
          // val yp = y - qt.t * (qt * y)

          val xx = (x dot x) - (qtx dot qtx) // = xp dot xp for xp = (I - q * q.t) x
          val xy = (x dot y) - (qtx dot qty) // = xp dot yp
          val yy = yyBc.value                // = yp dot yp


          val sign = if (xy > 0) 1 else -1
          val d = n - k - 2
          val chi2 = d * xy * xy / (xx * yy - xy * xy)
          val t = sign * math.sqrt(chi2)
          val b = xy / xx
          val se = b / t // = math.sqrt(|y - bx|^2 / (d * xx))

          LinRegOutput(nMissing, b, se, t)
        }
      }
    )
  }
}

case class LinearRegression(lr: RDD[(Variant, LinRegOutput)]) {

  def write(filename: String) {
    def toLine(v: Variant, lro: LinRegOutput) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt +
      "\t" + lro.nMissing + "\t" + lro.beta + lro.stdError + "\t" + lro.t
    lr.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tMISS\tBETA\tSE\tT\n")
  }
}
