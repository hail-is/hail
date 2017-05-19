package is.hail.methods

import breeze.linalg._
import breeze.numerics.sqrt
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T

object LinearRegressionMultiPheno {
  def schema = TStruct(
    ("beta", TArray(TDouble)),
    ("se", TArray(TDouble)),
    ("tstat", TArray(TDouble)),
    ("pval", TArray(TDouble)))

  def apply(vds: VariantDataset, ysExpr: Array[String], covExpr: Array[String], root: String, useDosages: Boolean, minAC: Int, minAF: Double): VariantDataset = {
    require(vds.wasSplit)

    val (y, cov, completeSamples) = RegressionUtils.getPhenosCovCompleteSamples(vds, ysExpr, covExpr)
    val sampleMask = vds.sampleIds.map(completeSamples.toSet).toArray

    val n = y.rows
    val k = cov.cols
    val d = n - k - 1
    val dRec = 1d / d

    if (minAC < 1)
      fatal(s"Minumum alternate allele count must be a positive integer, got $minAC")
    if (minAF < 0d || minAF > 1d)
      fatal(s"Minumum alternate allele frequency must lie in [0.0, 1.0], got $minAF")
    val combinedMinAC = math.max(minAC, (math.ceil(2 * n * minAF) + 0.5).toInt)

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linear regression for ${y.cols} ${ plural(y.cols, "phenotype") } on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast(y.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r))

    val yDummyBc = sc.broadcast(DenseVector.zeros[Double](n)) // dummy input in order to reuse RegressionUtils

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegressionMultiPheno.schema, pathVA)

    vds.mapAnnotations { case (v, va, gs) =>

      val (x: Vector[Double], isValid: Boolean) =
        if (useDosages) {
          val (x, mean) = RegressionUtils.dosageStats(gs, sampleMaskBc.value, n)
          (x, n * mean >= combinedMinAC)
        } else {
          val (x, nHet, nHomVar, nMissing) = RegressionUtils.hardCallStats(gs, sampleMaskBc.value)
          val ac = nHet + 2 * nHomVar
          (x, !(ac < combinedMinAC || ac == 2 * (n - nMissing) || (ac == (n - nMissing) && x.forall(_ == 1))))
        }

      val linregAnnot = if (isValid) {
        val qtx: DenseVector[Double] = QtBc.value * x
        val qty: DenseMatrix[Double] = QtyBc.value
        val xxpRec: Double = 1 / ((x dot x) - (qtx dot qtx))
        val xyp: DenseVector[Double] = (yBc.value.t * x) - (qty.t * qtx)
        val yyp: DenseVector[Double] = yypBc.value

        val b = xxpRec * xyp
        val se = sqrt(dRec * (xxpRec * yyp  - (b :* b)))
        val t = b :/ se
        val p = t.map(s => 2 * T.cumulative(-math.abs(s), d, true, false))

        Annotation(
          b.toArray: IndexedSeq[Double],
          se.toArray: IndexedSeq[Double],
          t.toArray: IndexedSeq[Double],
          p.toArray: IndexedSeq[Double])
      } else
        null

      val newAnnotation = inserter(va, linregAnnot)
      assert(newVAS.typeCheck(newAnnotation))
      newAnnotation
    }.copy[GenotypeMatrixT](vaSignature = newVAS)
  }
}