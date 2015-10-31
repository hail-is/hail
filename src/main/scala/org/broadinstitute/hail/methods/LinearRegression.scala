package org.broadinstitute.hail.methods

import java.io.File

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

import scala.collection.mutable
import scala.io.Source

case class CovariateData(sampleOfRow: Array[Int], covariateOfColumn: Array[String], data: DenseMatrix[Double])

object CovariateData {

  def read(filename: String, sampleIds: Array[String]): CovariateData = {
    val src = Source.fromFile(new File(filename))
    val header :: lines = src.getLines().filterNot(_.isEmpty).toList
    src.close()

    val covariateOfColumn = header.split("\\s+").tail
    val nCov = covariateOfColumn.length
    val nSamples = lines.length
    val sampleOfRow = Array.ofDim[Int](nSamples)
    val indexOfSample: Map[String, Int] = sampleIds.zipWithIndex.toMap

    val data = DenseMatrix.zeros[Double](nSamples, nCov)
    for (i <- 0 until nSamples) {
      val (sample, sampleCovs) = lines(i).split("\\s+").splitAt(1)
      sampleOfRow(i) = indexOfSample(sample(0))
      data(i to i, ::) := DenseVector(sampleCovs.map(_.toDouble))
    }
    CovariateData(sampleOfRow, covariateOfColumn, data)
  }
}

case class LinRegStats(nMissing: Int, beta: Double, se: Double, t: Double, p: Double)

class LinRegBuilder extends Serializable {
  var rowsX: mutable.ArrayBuilder.ofInt = new mutable.ArrayBuilder.ofInt()
  var valsX: mutable.ArrayBuilder.ofDouble = new mutable.ArrayBuilder.ofDouble()
  var sumX: Int = 0
  var sumXX: Int = 0
  var sumXY: Double = 0.0
  var missingRows: mutable.ArrayBuilder.ofInt = new mutable.ArrayBuilder.ofInt()

  def merge(row: Int, g: Genotype, y: DenseVector[Double]): LinRegBuilder = {
    g.call.map(_.gt) match {
      case Some(0) =>
      case Some(1) =>
        rowsX += row
        valsX += 1.0
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case Some(2) =>
        rowsX += row
        valsX += 2.0
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case None =>
        missingRows += row
    }
    this
  }

  def merge(that: LinRegBuilder): LinRegBuilder = {
    rowsX ++= that.rowsX.result()
    valsX ++= that.valsX.result()
    sumX += that.sumX
    sumXX += that.sumXX
    sumXY += that.sumXY
    missingRows ++= that.missingRows.result()

    this
  }
}

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    require(ped.phenoDefinedForAll)
    val rowOfSample = cov.sampleOfRow.zipWithIndex.toMap
    val isPhenotyped: Array[Boolean] = (0 until vds.nSamples).map(rowOfSample.isDefinedAt).toArray

    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    if (d < 1)
      throw new IllegalArgumentException(n + " samples and " + k + " covariates implies " + d + " degrees of freedom.")

    val sc = vds.sparkContext
    val rowOfSampleBc = sc.broadcast(rowOfSample)
    val isPhenotypedBc = sc.broadcast(isPhenotyped)
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val yArray = (0 until n).toArray.map(row => ped.phenoOf(cov.sampleOfRow(row)).toString.toDouble)
    val covAndOnesVector = DenseMatrix.horzcat(cov.data, DenseMatrix.ones[Double](n, 1))
    val y = DenseVector[Double](yArray)
    val qt = qr.reduced.justQ(covAndOnesVector).t
    val qty = qt * y

    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))

    new LinearRegression(vds
      .filterSamples(s => isPhenotypedBc.value(s))
      .aggregateByVariantWithKeys[LinRegBuilder](new LinRegBuilder())(
        (lrb, v, s, g) => lrb.merge(rowOfSampleBc.value(s),g, yBc.value),
        (lrb1, lrb2) => lrb1.merge(lrb2))
      .mapValues{ lrb =>
        assert(lrb.sumX > 0) //FIXME: better error handling

        val missingRows = lrb.missingRows.result()
        val nMissing = missingRows.size
        val meanX = lrb.sumX.toDouble / (n - nMissing)
        lrb.rowsX ++= missingRows
        lrb.valsX ++= Array.fill[Double](nMissing)(meanX)

        val rowsX = lrb.rowsX.result()
        val valsX = lrb.valsX.result()

        //SparseVector constructor expects sorted indices
        val indices = Array.range(0, rowsX.size)
        indices.sortBy(i => rowsX(i))
        val x = new SparseVector[Double](indices.map(rowsX(_)), indices.map(valsX(_)), n)

        val xx = lrb.sumXX + meanX * meanX * nMissing
        val xy = lrb.sumXY + meanX * missingRows.map(row => yBc.value(row)).sum
        val qtx = qtBc.value * x
        val qty = qtyBc.value
        val xxp = xx - (qtx dot qtx)
        val xyp = xy - (qtx dot qty)
        val yyp = yypBc.value

        val b = xyp / xxp
        val se = math.sqrt((yyp / xxp - b * b) / d)
        val t = b / se
        val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))

        LinRegStats(nMissing, b, se, t, p)
      }
    )
  }
}

case class LinearRegression(lr: RDD[(Variant, LinRegStats)]) {
  def write(filename: String) {
    def toLine(v: Variant, lrs: LinRegStats) = v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt +
      "\t" + lrs.nMissing + "\t" + lrs.beta + "\t" + lrs.se + "\t" + lrs.t + "\t" + lrs.p
    lr.map((toLine _).tupled)
      .writeTable(filename, "CHR\tPOS\tREF\tALT\tMISS\tBETA\tSE\tT\tP\n")
  }
}