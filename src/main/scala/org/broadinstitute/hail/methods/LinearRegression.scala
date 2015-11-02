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
    val linesIterator = src.getLines().filterNot(_.isEmpty)
    val header = linesIterator.next()
    val lines = linesIterator.toArray
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
  private val rowsX: mutable.ArrayBuilder.ofInt = new mutable.ArrayBuilder.ofInt()
  private val valsX: mutable.ArrayBuilder.ofDouble = new mutable.ArrayBuilder.ofDouble()
  private var sumX: Int = 0
  private var sumXX: Int = 0
  private var sumXY: Double = 0.0
  private val missingRows: mutable.ArrayBuilder.ofInt = new mutable.ArrayBuilder.ofInt()

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
      case _ => throw new IllegalArgumentException("Genotype value " + g.call.map(_.gt).get + " must be 0, 1, or 2.")
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

  def stats(y: DenseVector[Double], n: Int): (SparseVector[Double], Double, Double, Int) = {
    assert(sumX > 0) //FIXME: better error handling

    val missingRowsArray = missingRows.result()
    val nMissing = missingRowsArray.size
    val meanX = sumX.toDouble / (n - nMissing)
    rowsX ++= missingRowsArray
    (0 until nMissing).foreach(_ => valsX += meanX)

    val rowsXarray = rowsX.result()
    val valsXarray = valsX.result()

    //SparseVector constructor expects sorted indices
    val indices = Array.range(0, rowsXarray.size)
    indices.sortBy(i => rowsXarray(i))
    val x = new SparseVector[Double](indices.map(rowsXarray(_)), indices.map(valsXarray(_)), n)
    val xx = sumXX + meanX * meanX * nMissing
    val xy = sumXY + meanX * missingRowsArray.iterator.map(row => y(row)).sum

    (x, xx, xy, nMissing)
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

    val yArray = (0 until n).map(row => ped.phenoOf(cov.sampleOfRow(row)).toString.toDouble).toArray
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
        val (x, xx, xy, nMissing) = lrb.stats(yBc.value, n)

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