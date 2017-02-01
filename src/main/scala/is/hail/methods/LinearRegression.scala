package is.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.variant._

object LinearRegression {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset, ySA: String, covSA: Array[String], root: String, minAC: Int, minAF: Double): VariantDataset = {

    if (!vds.wasSplit)
      fatal("linreg requires bi-allelic VDS. Run split_multi or filter_multi first")

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

    info(s"Running linreg on $n samples with $k ${plural(k, "covariate")} including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegression.`type`, pathVA)

    vds.mapAnnotations{ case (v, va, gs) =>
      val lrb = new LinRegBuilder(yBc.value)
      gs.iterator.zipWithIndex.foreach { case (g, i) => if (sampleMaskBc.value(i)) lrb.merge(g) }

      val linRegAnnot = lrb.stats(yBc.value, n, combinedMinAC).map { stats =>
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

      inserter(va, linRegAnnot)
    }.copy(vaSignature = newVAS)
  }
}