package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import collection.mutable

case class LinRegStats(nMissing: Int, beta: Double, se: Double, t: Double, p: Double)

class LinRegBuilder extends Serializable {
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var sumX = 0
  private var sumXX = 0
  private val missingRows = new mutable.ArrayBuilder.ofInt()

  def merge(row: Int, g: Genotype): LinRegBuilder = {
    (g.gt: @unchecked) match {
      case Some(0) =>
      case Some(1) =>
        rowsX += row
        valsX += 1.0
        sumX += 1
        sumXX += 1
      case Some(2) =>
        rowsX += row
        valsX += 2.0
        sumX += 2
        sumXX += 4
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
    missingRows ++= that.missingRows.result()

    this
  }

  def stats(n: Int): Option[(SparseVector[Double], Double, Int)] = {
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

      Some((x, xx, nMissing))
    }
  }
}

object LinearRegression {

  def apply(vds: VariantDataset, ped: Pedigree, cov1: CovariateData): LinearRegression = {
    val cov = cov1.filterSamples(ped.phenotypedSamples)

    assert(cov.covRowSample.forall(ped.phenotypedSamples))

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
        (lrb, v, s, g) => lrb.merge(sampleCovRowBc.value(s), g),
        (lrb1, lrb2) => lrb1.merge(lrb2))
      .mapValues { lrb =>
        lrb.stats(n).map { stats => {
          val (x, xx, nMissing) = stats

          val qtx = qtBc.value * x
          val qty = qtyBc.value
          val xxp: Double = xx - (qtx dot qtx)
          val xyp: Double = (x dot yBc.value) - (qtx dot qty)
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

object LinearRegressionOnHcs {

  def apply(hcs: HardCallSet, ped: Pedigree, cov1: CovariateData): LinearRegression = {
    // FIXME: can remove filter and check for GoT2D
    val cov = cov1.filterSamples(ped.phenotypedSamples)

    if (!(hcs.localSamples sameElements cov.covRowSample))
      fatal("Samples misaligned, recreate .hcs using .ped and .cov")

    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    if (d < 1)
      throw new IllegalArgumentException(s"$n samples and $k covariates implies $d degrees of freedom.")

    info(s"Running linreg on $n samples and $k covariates...")

    val sc = hcs.sparkContext
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
        val GtVectorAndStats(x, xx, nMissing) = cs.hardStats(n)

        // FIXME: make condition more robust to rounding errors?
        // all HomRef | all Het | all HomVar
        if (xx == 0.0 || (x.size == n && xx == n) || xx == 4 * n)
          None
        else {
          val qtx = qtBc.value * x
          val qty = qtyBc.value
          val xxp: Double = xx - (qtx dot qtx)
          val xyp: Double = (x dot yBc.value) - (qtx dot qty)
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

case class LinearRegression(rdd: RDD[(Variant, Option[LinRegStats])]) {
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
    rdd.mapPartitions { it =>
      val sb = new StringBuilder
      it.map { case (v, olrs) => toLine(sb, v, olrs) }
    }.writeTable(filename, Some("CHR\tPOS\tREF\tALT\tPVAL\tBETA\tSE\tTSTAT\tMISS"))
  }
}