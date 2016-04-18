package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import scala.collection.mutable

object LinRegStats {
  def `type`: Type = TStruct(
    ("nMissing", TInt),
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))
}

case class LinRegStats(nMissing: Int, beta: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(nMissing, beta, se, t, p)
}

class LinRegBuilder extends Serializable {
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private val missingRows = new mutable.ArrayBuilder.ofInt()

  def merge(row: Int, g: Genotype, y: DenseVector[Double]): LinRegBuilder = {
    g.gt match {
      case Some(0) =>
      case Some(1) =>
        rowsX += row
        valsX += 1d
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case Some(2) =>
        rowsX += row
        valsX += 2d
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case None =>
        rowsX += row
        valsX += 0d // placeholder for meanX
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
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent
      missingRowsArray.foreach(row => valsXArray(row) = meanX)

      // rowsXArray is sorted, as expected by SparseVector constructor
      val x = new SparseVector[Double](rowsXArray, valsXArray, n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * missingRowsArray.iterator.map(y(_)).sum

      Some((x, xx, xy, nMissing))
    }
  }
}

object LinearRegression {
  def name = "LinearRegression"

  def apply(
    vds: VariantDataset,
    y: DenseVector[Double],
    cov: Option[DenseMatrix[Double]]): LinearRegression = {

    require(cov.forall(_.rows == y.size))

    val n = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n - k - 2

    if (d < 1)
      fatal(s"$n samples and $k covariates with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n samples with $k sample covariates...")

    val covAndOnes: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n, 1))
      case None => DenseMatrix.ones[Double](n, 1)
    }

    val qt = qr.reduced.justQ(covAndOnes).t
    val qty = qt * y

    val sc = vds.sparkContext
    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    // FIXME: worth making a version of aggregateByVariantWithKeys using sample index rather than sample name?
    val sampleIndexBc = sc.broadcast(vds.sampleIds.zipWithIndex.toMap)

    new LinearRegression(vds
      .aggregateByVariantWithKeys[LinRegBuilder](new LinRegBuilder())(
        (lrb, v, s, g) => lrb.merge(sampleIndexBc.value(s), g, yBc.value),
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

          LinRegStats(nMissing, b, se, t, p) }
        }
      }
    )
  }

  def apply(
    hcs: HardCallSet,
    y: DenseVector[Double],
    cov: Option[DenseMatrix[Double]]): LinearRegression = {

    require(cov.forall(_.rows == y.size))

    val n = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n - k - 2

    if (d < 1)
      fatal(s"$n samples and $k covariates with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n samples with $k sample covariates...")

    val covAndOnes: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n, 1))
      case None => DenseMatrix.ones[Double](n, 1)
    }

    val qt = qr.reduced.justQ(covAndOnes).t
    val qty = qt * y

    val sc = hcs.sparkContext
    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    new LinearRegression(hcs
      .rdd
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

case class LinearRegression(rdd: RDD[(Variant, Option[LinRegStats])])