package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T

object LinearRegressionBurden {
  def apply(vds0: VariantDataset, groupVariantsBy: String, aggregateWith: String, genotypeExpr: String, ySA: String, covSA: Array[String], minAC: Int, minAF: Double): KeyTable = {

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds0, ySA, covSA)

    val completeSamplesSet = completeSamples.toSet

    val keyExpr = groupVariantsBy
    val key = groupVariantsBy.takeWhile(_ != "=").trim

    val aggExpr = completeSamples.map(s => s"$s = $s.$aggregateWith").mkString(", ")

    val kt = vds0.filterSamples { case (s, sa) => completeSamplesSet(s) }
      .makeKT(keyExpr, s"`` = $aggregateWith", Array(key))
      .explode(key)
      .aggregate(s"$key = $key", aggExpr)

    kt
//    val n = y.size
//    val k = cov.cols
//    val d = n - k - 1
//
//    if (minAC < 1)
//      fatal(s"Minumum alternate allele count must be a positive integer, got $minAC")
//    if (minAF < 0d || minAF > 1d)
//      fatal(s"Minumum alternate allele frequency must lie in [0.0, 1.0], got $minAF")
//    val combinedMinAC = math.max(minAC, (math.ceil(2 * n * minAF) + 0.5).toInt)
//
//    if (d < 1)
//      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")
//
//    info(s"Running linreg on $n samples with $k ${ plural(k, "covariate") } including intercept...")
//
//    val Qt = qr.reduced.justQ(cov).t
//    val Qty = Qt * y
//
//    val sc = vds.sparkContext
//    val sampleMaskBc = sc.broadcast(sampleMask)
//    val yBc = sc.broadcast(y)
//    val QtBc = sc.broadcast(Qt)
//    val QtyBc = sc.broadcast(Qty)
//    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))
//
//    vds.mapAnnotations { case (v, va, gs) =>
//      val lrb = new LinRegBuilder(yBc.value)
//      val gts = gs.hardCallIterator
//      val mask = sampleMaskBc.value
//      var i = 0
//      while (i < mask.length) {
//        val gt = gts.nextInt()
//        if (mask(i))
//          lrb.merge(gt)
//        i += 1
//      }
//
//      val linregAnnot = lrb.stats(yBc.value, n, combinedMinAC).map { stats =>
//        val (x, xx, xy) = stats
//
//        val qtx = QtBc.value * x
//        val qty = QtyBc.value
//        val xxp: Double = xx - (qtx dot qtx)
//        val xyp: Double = xy - (qtx dot qty)
//        val yyp: Double = yypBc.value
//
//        val b = xyp / xxp
//        val se = math.sqrt((yyp / xxp - b * b) / d)
//        val t = b / se
//        val p = 2 * T.cumulative(-math.abs(t), d, true, false)
//
//        Annotation(b, se, t, p)
//      }
//      .orNull
//    }
  }
}