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

case class LinRegStat(nMissing: Int, beta: Double, se: Double, t: Double, p: Double)

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    require(ped.phenoDefinedForAll)
    val rowOfSample = cov.sampleOfRow.zipWithIndex.toMap
    val isPhenotyped: Array[Boolean] = (0 until vds.nSamples).map(rowOfSample.isDefinedAt).toArray
    
    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    require(d > 0)
    
    val sc = vds.sparkContext
    val rowOfSampleBc = sc.broadcast(rowOfSample)
    val isPhenotypedBc = sc.broadcast(isPhenotyped)
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val yArray = (0 until n).toArray.map(row => ped.phenoOf(cov.sampleOfRow(row)).toString.toDouble)
    val covAndOnesVector = DenseMatrix.horzcat(cov.data, DenseMatrix.ones[Double](n, 1))
    val y = DenseVector[Double](yArray)
    val qt = qr.reduced.justQ(covAndOnesVector).t
    val qty = qt * y
    
    val yBc =   sc.broadcast(y)
    val qtBc =  sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))
    
    type LinRegBuilder = (List[(Int, Int)], Int, Int, Double, List[Int])

    def zeroValue: LinRegBuilder = (List(), 0, 0, 0.0, List())

    def seqOp(l: LinRegBuilder, rowGt: (Int, Option[Int])): LinRegBuilder = {
      val (sparseX, sumX, sumXX, sumXY, missRows) = l
      val (row, gt) = rowGt
      gt match {
        case Some(0) => (                 sparseX, sumX    , sumXX    , sumXY                     ,        missRows)
        case Some(1) => ((row, gt.get) :: sparseX, sumX + 1, sumXX + 1, sumXY +     yBc.value(row),        missRows)
        case Some(2) => ((row, gt.get) :: sparseX, sumX + 2, sumXX + 4, sumXY + 2 * yBc.value(row),        missRows)
        case None    => (                 sparseX, sumX    , sumXX    , sumXY                     , row :: missRows)
      }
    }

    def combOp(l1: LinRegBuilder, l2: LinRegBuilder): LinRegBuilder =
      (l1._1 ++ l2._1, l1._2 + l2._2, l1._3 + l2._3, l1._4 + l2._4, l1._5 ++ l2._5)

    new LinearRegression(vds
      .filterSamples(s => isPhenotypedBc.value(s))
      .mapWithKeys[(Variant, (Int, Option[Int]))]{ (v, s, g) => (v, (rowOfSampleBc.value(s), g.call.map(_.gt))) }
      .aggregateByKey[LinRegBuilder](zeroValue)(seqOp, combOp)
      .mapValues{ case (sparseX, sumX, sumXX, sumXY, missRows) =>
        assert(sumX > 0) //FIXME: better error handling

        val (rows, gts) = sparseX.sortBy(_._1).unzip
        val x = new SparseVector[Double](rows.toArray, gts.map(_.toDouble).toArray, n)

        val nMiss = missRows.length
        val meanX = sumX.toDouble / (n - nMiss)
        missRows.foreach(row => x(row) = meanX)

        val xx = sumXX + meanX * meanX * nMiss
        val xy = sumXY + meanX * missRows.map(row => yBc.value(row)).sum

        val qtx = qtBc.value * x
        val qty = qtyBc.value

        val xxp = xx - (qtx dot qtx)
        val xyp = xy - (qtx dot qty)
        val yyp = yypBc.value

        val b = xyp / xxp
        val se = math.sqrt((yyp / xxp - b * b) / d)
        val t = b / se
        val p = 2 * tDistBc.value.cumulativeProbability(- math.abs(t))

        LinRegStat(nMiss, b, se, t, p)
      }
    )
  }
}

case class LinearRegression(lr: RDD[(Variant, LinRegStat)]) {

  def write(filename: String) {
    def toLine(v: Variant, lrs: LinRegStat) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt +
      "\t" + lrs.nMissing + "\t" + lrs.beta + "\t" + lrs.se + "\t" + lrs.t + "\t" + lrs.p
    lr.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tMISS\tBETA\tSE\tT\tP\n")
  }
}
