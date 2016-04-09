package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import scala.collection.mutable.ArrayBuffer

object LinRegStats {
  def `type`: Type = TStruct(("nMissing", TInt), ("beta", TDouble),
    ("se", TDouble), ("tstat", TDouble), ("pval", TDouble))
}

case class LinRegStats(nMissing: Int, beta: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(nMissing, beta, se, t, p)
}

class LinRegBuilder extends Serializable {
  private val rowsX = ArrayBuffer[Int]()
  private val valsX = ArrayBuffer[Double]()
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private val missingRows = ArrayBuffer[Int]()

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
    val missingRowsArray = missingRows.toArray
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
      val x = new SparseVector[Double](rowsX.toArray, valsX.toArray, n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * missingRowsArray.iterator.map(y(_)).sum

      Some((x, xx, xy, nMissing))
    }
  }
}

object LinearRegression {
  def name = "LinearRegression"

  def apply(vds: VariantDataset, y: DenseVector[Double], cov: Option[DenseMatrix[Double]]): LinearRegression = {
    require(cov.forall(_.rows == y.size))

    val n = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n - k - 2

    if (d < 1)
      fatal(s"$n samples and $k covariates with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n samples and $k covariates...")

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

    // FIXME: remove once localSamples is gone
    val remapSamplesBc = sc.broadcast(vds.localSamples.zipWithIndex.toMap)

    new LinearRegression(vds
      .aggregateByVariantWithKeys[LinRegBuilder](new LinRegBuilder())(
        (lrb, v, s, g) => lrb.merge(remapSamplesBc.value(s), g, yBc.value),
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
}

case class LinearRegression(rdd: RDD[(Variant, Option[LinRegStats])]) {
  def write(filename: String) {
    def toLine(v: Variant, olrs: Option[LinRegStats]) = olrs match {
      case Some(lrs) => v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\t" + lrs.nMissing + "\t" + lrs.beta + "\t" + lrs.se + "\t" + lrs.t + "\t" + lrs.p
      case None => v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\tNA\tNA\tNA\tNA\tNA"
    }
    rdd.map((toLine _).tupled)
      .writeTable(filename, Some("CHR\tPOS\tREF\tALT\tMISS\tBETA\tSE\tT\tP"))
  }
}