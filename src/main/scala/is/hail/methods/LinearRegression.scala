package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T

object LinearRegression {
  val schema: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset, ySA: String, covSA: Array[String], root: String, useDosages: Boolean, useDominance: Boolean, useNormalise: Boolean, minAC: Int, minAF: Double): VariantDataset = {
    require(vds.wasSplit)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, ySA, covSA)
    val sampleMask = vds.sampleIds.map(completeSamples.toSet).toArray

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (minAC < 1)
      fatal(s"Minumum alternate allele count must be a positive integer, got $minAC")
    if (minAF < 0d || minAF > 1d)
      fatal(s"Minumum alternate allele frequency must lie in [0.0, 1.0], got $minAF")
    val combinedMinAC = math.max(minAC, (math.ceil(2 * n * minAF) + 0.5).toInt)

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linear regression on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))
    val yy = y dot y
    val ymean = sum(y) / n

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegression.schema, pathVA)

    vds.mapAnnotations { case (v, va, gs) =>

      val optStats: Option[(Vector[Double], Double, Double, Double)] =
        if (useDosages)
          RegressionUtils.toLinregDosageStats(gs, yBc.value, sampleMaskBc.value, combinedMinAC)
        else if (useDominance)
          RegressionUtils.toLinregDominanceHardCallStats(gs, yBc.value, sampleMaskBc.value, combinedMinAC)
        else
          RegressionUtils.toLinregHardCallStats(gs, yBc.value, sampleMaskBc.value, combinedMinAC)

      val linregAnnot = optStats.map { stats =>
        val (x, xx, xy, xmean) = stats

        val qtx = QtBc.value * x
        val qty = QtyBc.value
        val xxp: Double = xx - (qtx dot qtx)
        val xyp: Double = xy - (qtx dot qty)
        val yyp: Double = yypBc.value

        val scale = if (useNormalise) {
          val xscale: Double = math.sqrt(xx - xmean * xmean * n)
          val yscale: Double = math.sqrt(yy - ymean * ymean * n)
          xscale / yscale
        } else {
          1
        }

        val b_unscaled = xyp / xxp
        val b = scale * b_unscaled
        val se = scale * math.sqrt((yyp / xxp - b_unscaled * b_unscaled) / d)
        val t = b / se
        val p = 2 * T.cumulative(-math.abs(t), d, true, false)

        Annotation(b, se, t, p)
      }
      .orNull

      val newAnnotation = inserter(va, linregAnnot)
      assert(newVAS.typeCheck(newAnnotation))
      newAnnotation
    }.copy(vaSignature = newVAS)
  }
}