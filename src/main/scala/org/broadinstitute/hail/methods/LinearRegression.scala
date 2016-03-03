package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

case class LinRegStats(nMissing: Int, beta: Double, se: Double, t: Double, p: Double)

class LinRegBuilder extends Serializable {
  private val rowsX = new collection.mutable.ArrayBuilder.ofInt()
  private val valsX = new collection.mutable.ArrayBuilder.ofDouble()
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private val missingRows = new collection.mutable.ArrayBuilder.ofInt()

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

  def stats(y: DenseVector[Double], n: Int): Option[(SparseVector[Double], Double, Double, Int)] = {
    val missingRowsArray = missingRows.result()
    val nMissing = missingRowsArray.size
    val nPresent = n - nMissing

    // all HomRef | all Het | all HomVar
    if (sumX == 0 || (sumX == nPresent && sumXX == nPresent) || sumX == 2 * nPresent)
      None
    else {
      val meanX = sumX.toDouble / nPresent
      rowsX ++= missingRowsArray
      (0 until nMissing).foreach(_ => valsX += meanX)

      //SparseVector constructor expects sorted indices, follows from sorting of covRowSample
      val x = new SparseVector[Double](rowsX.result(), valsX.result(), n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * missingRowsArray.iterator.map(y(_)).sum

      Some((x, xx, xy, nMissing))
    }
  }
}

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, ped: Pedigree, cov: CovariateData): LinearRegression = {
    // LinearRegressionCommand uses cov.filterSamples(ped.phenotypedSamples) in call
    require(cov.covRowSample.forall(ped.phenotypedSamples))

    val sampleCovRow = cov.covRowSample.zipWithIndex.toMap

    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    if (d < 1)
      throw new IllegalArgumentException(s"$n samples and $k covariates implies $d degrees of freedom.")

    info(s"Running linreg on $n samples and $k covariates...")

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
      .filterSamples { case (s, sa) => samplesWithCovDataBc.value(s) }
      .aggregateByVariantWithKeys[LinRegBuilder](new LinRegBuilder())(
        (lrb, v, s, g) => lrb.merge(sampleCovRowBc.value(s), g, yBc.value),
        (lrb1, lrb2) => lrb1.merge(lrb2))
      .mapValues { lrb =>
        lrb.stats(yBc.value, n).map { stats => {
          val (x, xx, xy, nMissing) = stats

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
        }
      }
    )
  }
}

object LinearRegressionFromHardCallSet {
  def name = "LinearRegressionFromHardCallSet"

  def apply(hcs: HardCallSet, ped: Pedigree, cov: CovariateData): LinearRegression = {

    // LinearRegressionFromHardCallSetCommand uses cov.filterSamples(ped.phenotypedSamples) in call
    require(cov.covRowSample.forall(ped.phenotypedSamples)) // FIXME: Code below assumes same samples in hcs and cov, true for GoT2D

    //val sampleCovRow = cov.covRowSample.zipWithIndex.toMap

    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    if (d < 1)
      throw new IllegalArgumentException(s"$n samples and $k covariates implies $d degrees of freedom.")

    info(s"Running linreg on $n samples and $k covariates...")

    val sc = hcs.sparkContext
    //val sampleCovRowBc = sc.broadcast(sampleCovRow)
    //val samplesWithCovDataBc = sc.broadcast(sampleCovRow.keySet)
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

    new LinearRegression(hcs.rdd
      .mapValues { cs =>
        val GtVectorAndStats(x, xx, xy, nMissing) = cs.hardStats(yBc.value, n)

        // FIXME: make condition more robust to rounding errors?
        // all HomRef | all Het | all HomVar
        if (xx == 0.0 || (x.size == n && xx == n) || xx == 4 * n)
          None
        else {
          val qtx = qtBc.value * x
          val qty = qtyBc.value
          val xxp: Double = xx - (qtx dot qtx)
          val xyp: Double = xy - (qtx dot qty)
          val yyp: Double = yypBc.value

          val b: Double = xyp / xxp
          val se = math.sqrt((yyp / xxp - b * b) / d)
          val t = b / se
          val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))

          Some(LinRegStats(nMissing, b, se, t, p))
        }
      }
    )
  }
}

case class LinearRegression(lr: RDD[(Variant, Option[LinRegStats])]) {
  def write(filename: String) {
    def toLine(sb: StringBuilder, v: Variant, olrs: Option[LinRegStats]): String = {
      sb.clear()
      olrs match {
        case Some(lrs) =>
          sb.tsvAppendItems(v.contig, v.start, v.ref, v.alt, lrs.p, lrs.beta, lrs.se, lrs.t, lrs.nMissing)
        case None =>
          sb.tsvAppendItems(v.contig, v.start, v.ref, v.alt, "NA\tNA\tNA\tNA\tNA")
      }
      sb.result()
    }
    lr.mapPartitions { it =>
      val sb = new StringBuilder
      it.map { case (v, olrs) => toLine(sb, v, olrs) }
    }.writeTable(filename, Some("CHR\tPOS\tREF\tALT\tPVAL\tBETA\tSE\tTSTAT\tMISS"))
  }
}