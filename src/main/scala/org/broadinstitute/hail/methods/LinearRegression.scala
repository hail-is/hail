package org.broadinstitute.hail.methods

import java.io.File

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

import scala.collection.mutable
import scala.io.Source

case class CovariateData(covRowSample: Array[Int], covName: Array[String], data: DenseMatrix[Double])

object CovariateData {

  def read(filename: String, sampleIds: IndexedSeq[String]): CovariateData = {
    val src = Source.fromFile(new File(filename))
    val linesIt = src.getLines().filterNot(_.isEmpty)
    val header = linesIt.next()
    val lines = linesIt.toArray
    src.close()

    val covName = header.split("\\s+").tail
    val nCov = covName.length
    val nCovRow = lines.length
    val covRowSample = Array.ofDim[Int](nCovRow)
    val sampleNameIndex: Map[String, Int] = sampleIds.zipWithIndex.toMap

    val data = DenseMatrix.zeros[Double](nCovRow, nCov)
    for (cr <- 0 until nCovRow) {
      val entries = lines(cr).split("\\s+")
      covRowSample(cr) = sampleNameIndex(entries(0))
      data(cr to cr, ::) := DenseVector(entries.iterator.drop(1).map(_.toDouble).toArray)
    }
    CovariateData(covRowSample, covName, data)
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
    g.gt match {
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
      case _ => throw new IllegalArgumentException("Genotype value " + g.gt.get + " must be 0, 1, or 2.")
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

    val rowsXArray = rowsX.result()
    val valsXArray = valsX.result()

    //SparseVector constructor expects sorted indices
    val indices = Array.range(0, rowsXArray.size)
    indices.sortBy(i => rowsXArray(i))
    val x = new SparseVector[Double](indices.map(rowsXArray(_)), indices.map(valsXArray(_)), n)
    val xx = sumXX + meanX * meanX * nMissing
    val xy = sumXY + meanX * missingRowsArray.iterator.map(y(_)).sum

    (x, xx, xy, nMissing)
  }
}

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    require(ped.trios.forall(_.pheno.isDefined))
    val sampleCovRow = cov.covRowSample.zipWithIndex.toMap

    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    if (d < 1)
      throw new IllegalArgumentException(n + " samples and " + k + " covariates implies " + d + " degrees of freedom.")

    val sc = vds.sparkContext
    val sampleCovRowBc = sc.broadcast(sampleCovRow)
    val samplesWithCovDataBc = sc.broadcast(sampleCovRow.keySet)
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val samplePheno = ped.samplePheno
    val yArray = (0 until n).flatMap(cr => samplePheno(cov.covRowSample(cr)).map(_.toString.toDouble)).toArray
    val covAndOnesVector = DenseMatrix.horzcat(cov.data, DenseMatrix.ones[Double](n, 1))
    val y = DenseVector[Double](yArray)
    val qt = qr.reduced.justQ(covAndOnesVector).t
    val qty = qt * y

    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))

    new LinearRegression(vds
      .filterSamples(samplesWithCovDataBc.value.contains)
      .aggregateByVariantWithKeys[LinRegBuilder](new LinRegBuilder())(
        (lrb, v, s, g) => lrb.merge(sampleCovRowBc.value(s), g, yBc.value),
        (lrb1, lrb2) => lrb1.merge(lrb2))
      .mapValues{ lrb =>
        val (x, xx, xy, nMissing) = lrb.stats(yBc.value, n)

        val qtx = qtBc.value * x
        val qty = qtyBc.value
        val xxp: Double = xx - (qtx dot qtx)
        val xyp: Double = xy - (qtx dot qty)
        val yyp: Double = yypBc.value

        val b: Double = xyp / xxp
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