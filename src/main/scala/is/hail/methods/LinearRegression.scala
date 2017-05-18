package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T

object LinearRegression {
  val schema = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset, yExpr: String, covExpr: Array[String], root: String, useDosages: Boolean, minAC: Int, minAF: Double): VariantDataset = {
    require(vds.wasSplit)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
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

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegression.schema, pathVA)

    vds.mapAnnotations { case (v, va, gs) =>

      val (x: Vector[Double], isValid: Boolean) =
        if (useDosages) {
          val (x, mean) = RegressionUtils.dosageStats(gs, Some(sampleMaskBc.value), n)
          (x, n * mean >= combinedMinAC)
        } else {
          val (x, nHet, nHomVar, nMissing) = RegressionUtils.hardCallStats(gs, Some(sampleMaskBc.value))
          val ac = nHet + 2 * nHomVar
          (x, !(ac < combinedMinAC || ac == 2 * (n - nMissing) || (ac == (n - nMissing) && x.forall(_ == 1))))
        }

      val linregAnnot = if (isValid) {
        val qtx = QtBc.value * x
        val qty = QtyBc.value
        val xxp: Double = (x dot x) - (qtx dot qtx)
        val xyp: Double = (x dot y) - (qtx dot qty)
        val yyp: Double = yypBc.value

        val b = xyp / xxp
        val se = math.sqrt((yyp / xxp - b * b) / d)
        val t = b / se
        val p = 2 * T.cumulative(-math.abs(t), d, true, false)

        Annotation(b, se, t, p)
      } else
        null

      val newAnnotation = inserter(va, linregAnnot)
      assert(newVAS.typeCheck(newAnnotation))
      newAnnotation
    }.copy(vaSignature = newVAS)
  }
}