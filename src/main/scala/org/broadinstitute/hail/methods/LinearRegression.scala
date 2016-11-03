package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import scala.collection.mutable

class LinRegBuilder(y: DenseVector[Double]) extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var row = 0
  private var sparseLength = 0 // length of rowsX and valsX (ArrayBuilder has no length), used to track missingRowIndices
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private var sumYMissing = 0.0

  def merge(g: Genotype): LinRegBuilder = {
    (g.unboxedGT: @unchecked) match {
      case 0 =>
      case 1 =>
        rowsX += row
        valsX += 1d
        sparseLength += 1
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case 2 =>
        rowsX += row
        valsX += 2d
        sparseLength += 1
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
        sumYMissing += y(row)
    }
    row += 1

    this
  }

  def stats(y: DenseVector[Double], n: Int, minAC: Int): Option[(SparseVector[Double], Double, Double)] = {
    require(minAC > 0)

    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = n - nMissing
    val allHet = sumX == nPresent && sumXX == nPresent
    val allHomVar = sumX == 2 * nPresent

    if (sumX < minAC || allHomVar || allHet)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      val x = new SparseVector[Double](rowsXArray, valsXArray, n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some((x, xx, xy))
    }
  }
}

object LinearRegression {
  def schema: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset, pathVA: List[String], completeSamples: IndexedSeq[String], y: DenseVector[Double], cov: Option[DenseMatrix[Double]], minAC: Int): VariantDataset = {
    require(cov.forall(_.rows == y.size))

    val n = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n - k - 2

    if (d < 1)
      fatal(s"$n samples and $k ${plural(k, "covariate")} with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n samples with $k sample ${plural(k, "covariate")}...")

    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray

    val (newVAS, inserter) = vds.insertVA(LinearRegression.schema, pathVA)

    val covAndOnes: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n, 1))
      case None => DenseMatrix.ones[Double](n, 1)
    }

    val Qt = qr.reduced.justQ(covAndOnes).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    vds.mapAnnotations{ case (v, va, gs) =>
      val lrb = new LinRegBuilder(yBc.value)
      gs.iterator.zipWithIndex.foreach { case (g, i) => if (sampleMaskBc.value(i)) lrb.merge(g) }

      val linRegStat = lrb.stats(yBc.value, n, minAC).map { stats =>
        val (x, xx, xy) = stats

        val qtx = QtBc.value * x
        val qty = QtyBc.value
        val xxp: Double = xx - (qtx dot qtx)
        val xyp: Double = xy - (qtx dot qty)
        val yyp: Double = yypBc.value

        val b = xyp / xxp
        val se = math.sqrt((yyp / xxp - b * b) / d)
        val t = b / se
        val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))

        Annotation(b, se, t, p)
      }

      inserter(va, linRegStat)
    }.copy(vaSignature = newVAS)
  }
}

object LinearRegressionHcs {
  def apply(
    hcs: HardCallSet,
    y: DenseVector[Double],
    cov: Option[DenseMatrix[Double]],
    sampleFilter: Array[Boolean],
    reduceSampleIndex: Array[Int],
    minMAC: Int = 0,
    maxMAC: Int = Int.MaxValue): LinearRegressionHcs = {

    val n = hcs.nSamples
    val n0 = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n0 - k - 2

    assert(n >= n0)
    assert(cov.forall(_.rows == n0))

    if (d < 1)
      fatal(s"$n0 samples and $k ${plural(k, "covariate")} with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n0 samples with $k sample ${plural(k, "covariate")}...")

    val covAndOnes: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n0, 1))
      case None => DenseMatrix.ones[Double](n0, 1)
    }

    val qt = qr.reduced.justQ(covAndOnes).t
    val qty = qt * y

    val sc = hcs.sparkContext
    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val sampleFilterBc = sc.broadcast(sampleFilter)
    val reduceSampleIndexBc = sc.broadcast(reduceSampleIndex)

    new LinearRegressionHcs(hcs
      .rdd
      .mapValues { cs => // FIXME: only three are necessary
        val GtVectorAndStats(x, nHomRef, nHet, nHomVar, nMissing, meanX) = cs.hardStats(n, n0, sampleFilterBc.value, reduceSampleIndexBc.value)

        val nPresent = n0 - nMissing
        val mac = nHet + nHomVar * 2

        if (nHomRef == nPresent || nHet == nPresent || nHomVar == nPresent || mac < minMAC || mac > maxMAC)
          None
        else {
          val xx = nHet + nHomVar * 4 + nMissing * meanX * meanX
          val qtx = qtBc.value * x
          val qty = qtyBc.value
          val xxp: Double = xx - (qtx dot qtx)
          val xyp: Double = (x dot yBc.value) - (qtx dot qty)
          val yyp: Double = yypBc.value

          if (xxp == 0d)
            None
          else {
            val b: Double = xyp / xxp
            val se = math.sqrt((yyp / xxp - b * b) / d)
            if (se == 0)
              None
            else {
              val t = b / se
              val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))
              if (p.isNaN)
                None
              else
                Some(LinRegStatsHcs(b, se, t, p))
            }
          }
        }
      }
    )
  }
}

case class LinearRegressionHcs(rdd: RDD[(Variant, Option[LinRegStatsHcs])])

object LinRegStatsHcs {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))
}

case class LinRegStatsHcs(beta: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(beta, se, t, p)
}
