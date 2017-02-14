package is.hail.methods

import breeze.linalg._
import breeze.stats.mean
import is.hail.annotations._
import is.hail.expr.{TDouble, TStruct}
import is.hail.stats._
import is.hail.utils._
import is.hail.variant.Variant
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class LinearMixedRegressionSuite extends SparkSuite {

  @Test def lmmSmallExampleTest() {

    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val C = DenseMatrix(
      (1.0, 0.0, -1.0),
      (1.0, 2.0, 3.0),
      (1.0, 1.0, 5.0),
      (1.0, -2.0, 0.0),
      (1.0, -2.0, -4.0),
      (1.0, 4.0, 3.0))

    val G = DenseMatrix(
      (0, 1, 1, 2),
      (1, 0, 2, 2),
      (2, 0, 0, 2),
      (0, 0, 1, 1),
      (0, 0, 0, 1),
      (2, 1, 0, 0))

    val n = y.length
    val c = C.cols

    val W = convert(G(::, 0 to 1), Double)

    // each row has mean 0, norm sqrt(n), variance 1
    for (i <- 0 until W.cols) {
      W(::, i) -= mean(W(::, i))
      W(::, i) *= math.sqrt(n) / norm(W(::, i))
    }

    val mW = W.cols
    val mG = G.cols

    val rrm = (W * W.t) / mW.toDouble // RRM
    val delta = 2.23

    // Now testing global model
    // First solve directly with Cholesky
    val V = rrm + DenseMatrix.eye[Double](n) * delta

    val invChol = inv(cholesky(V))

    val yc = invChol * y
    val Cc = invChol * C

    val beta = (Cc.t * Cc) \ (Cc.t * yc)
    val res = norm(yc - Cc * beta)
    val sg2 = (res * res) / (n - c)
    val se2 = delta * sg2
    val h2 = sg2 / (se2 + sg2)

    // Then solve with DiagLMM and compare
    val eigRRM = eigSymD(rrm)
    val Ut = eigRRM.eigenvectors.t
    val S = eigRRM.eigenvalues

    val yr = Ut * y
    val Cr = Ut * C

    val model = DiagLMM(Cr, yr, S, Some(delta))

    TestUtils.assertVectorEqualityDouble(beta, model.globalB)
    assert(D_==(sg2, model.globalS2))

    val modelML = DiagLMM(Cr, yr, S, Some(delta), useML = true)

    TestUtils.assertVectorEqualityDouble(beta, modelML.globalB)
    assert(D_==(sg2 * (n - c) / n, modelML.globalS2))

    // Now testing association per variant
    // First solve directly with Cholesky
    val directResult = (0 until mG).map { j =>
      val xIntArray = G(::, j).toArray
      val x = convert(G(::, j to j), Double)
      val xC = DenseMatrix.horzcat(x, C)
      val xCc = invChol * xC
      val beta = (xCc.t * xCc) \ (xCc.t * yc)
      val res = norm(yc - xCc * beta)
      val sg2 = (res * res) / (n - c)
      val chi2 = n * (model.logNullS2- math.log(sg2))
      val pval = chiSquaredTail(1d, chi2)
      val nHomRef = xIntArray.count(_ == 0)
      val nHet = xIntArray.count(_ == 1)
      val nHomVar = xIntArray.count(_ == 2)
      val nMissing = xIntArray.count(_ == -1)
      val af = (nHet + 2 * nHomVar).toDouble / (2 * (n - nMissing))
      (Variant("1", j + 1, "A", "C"), (beta(0), sg2, chi2, pval, af, nHomRef, nHet, nHomVar, nMissing))
    }.toMap

    // Then solve with LinearMixeModel and compare
    val vds0 = vdsFromMatrix(sc)(G)
    val pheno = y.toArray
    val cov1 = C(::, 1).toArray
    val cov2 = C(::, 2).toArray

    val assocVds = vds0
      .annotateSamples(vds0.sampleIds.zip(pheno).toMap, TDouble, "sa.pheno")
      .annotateSamples(vds0.sampleIds.zip(cov1).toMap, TDouble, "sa.cov1")
      .annotateSamples(vds0.sampleIds.zip(cov2).toMap, TDouble, "sa.cov2")
    val kinshipVds = assocVds.filterVariants((v, va, gs) => v.start <= 2)

    val vds = LinearMixedRegression(assocVds, kinshipVds, "sa.pheno", covSA = Array("sa.cov1", "sa.cov2"),
      useML = false, rootGA = "global.lmmreg", rootVA = "va.lmmreg", runAssoc = true, optDelta = Some(delta),
      sparsityThreshold = 1.0, forceBlock = false, forceGrammian = false)

    val qBeta = vds.queryVA("va.lmmreg.beta")._2
    val qSg2 = vds.queryVA("va.lmmreg.sigmaG2")._2
    val qChi2 = vds.queryVA("va.lmmreg.chi2")._2
    val qPval = vds.queryVA("va.lmmreg.pval")._2
    val qAF = vds.queryVA("va.lmmreg.AF")._2
    val qnHomRef = vds.queryVA("va.lmmreg.nHomRef")._2
    val qnHet = vds.queryVA("va.lmmreg.nHet")._2
    val qnHomVar = vds.queryVA("va.lmmreg.nHomVar")._2
    val qnMissing = vds.queryVA("va.lmmreg.nMissing")._2

    val annotationMap = vds.variantsAndAnnotations.collect().toMap

    def assertInt(q: Querier, v: Variant, value: Int) = q(annotationMap(v)).get.asInstanceOf[Int] == value

    def assertDouble(q: Querier, v: Variant, value: Double) = {
      val x = q(annotationMap(v)).get.asInstanceOf[Double]
      assert(D_==(x, value))
    }

    (0 until mG).foreach { j =>
      val v = Variant("1", j + 1, "A", "C")
      val (beta, sg2, chi2, pval, af, nHomRef, nHet, nHomVar, nMissing) = directResult(v)
      assertDouble(qBeta, v, beta)
      assertDouble(qSg2, v, sg2)
      assertDouble(qChi2, v, chi2)
      assertDouble(qPval, v, pval)
      assertDouble(qAF, v, af)
      assertInt(qnHomRef, v, nHomRef)
      assertInt(qnHet, v, nHet)
      assertInt(qnHomVar, v, nHomVar)
      assertInt(qnMissing, v, nMissing)
    }
  }

  @Test def lmmLargeExampleTest() {
    val seed = 0
    scala.util.Random.setSeed(seed)

    val n = 100
    val c = 3 // number of covariates including intercept
    val m0 = 300
    val k = 10
    val Fst = .2

    val y = DenseVector.fill[Double](n)(scala.util.Random.nextGaussian())

    val C =
      if (c == 1)
        DenseMatrix.ones[Double](n, 1)
      else
        DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(scala.util.Random.nextGaussian()))

    val FstOfPop = Array.fill[Double](k)(Fst)

    val bnm = BaldingNicholsModel(sc, k, n, m0, None, Some(FstOfPop), scala.util.Random.nextInt(), Some(4), UniformDist(.1, .9))

    val G = TestUtils.removeConstantCols(TestUtils.vdsToMatrixInt(bnm))

    val mG = G.cols
    val mW = G.cols

    // println(s"$mG of $m0 variants are not constant")

    val W = convert(G(::, 0 until mW), Double)

    // each row has mean 0, norm sqrt(n), variance 1
    // each row has mean 0, norm sqrt(n), variance 1
    for (i <- 0 until mW) {
      W(::, i) -= mean(W(::, i))
      W(::, i) *= math.sqrt(n) / norm(W(::, i))
    }

    val rrm = (W * W.t) / mW.toDouble // RRM
    val delta = scala.util.Random.nextGaussian()

    // Now testing global model
    // First solve directly with Cholesky
    val V = rrm + DenseMatrix.eye[Double](n) * delta

    val invChol = inv(cholesky(V))

    val yc = invChol * y
    val Cc = invChol * C

    val beta = (Cc.t * Cc) \ (Cc.t * yc)
    val res = norm(yc - Cc * beta)
    val sg2 = (res * res) / (n - c)
    val se2 = delta * sg2
    val h2 = sg2 / (se2 + sg2)

    // println(beta, sg2, se2, h2)

    // Then solve with DiagLMM and compare
    val eigRRM = eigSymD(rrm)
    val Ut = eigRRM.eigenvectors.t
    val S = eigRRM.eigenvalues

    val yr = Ut * y
    val Cr = Ut * C

    val model = DiagLMM(Cr, yr, S, Some(delta))

    TestUtils.assertVectorEqualityDouble(beta, model.globalB)
    assert(D_==(sg2, model.globalS2))

    val modelML = DiagLMM(Cr, yr, S, Some(delta), useML = true)

    TestUtils.assertVectorEqualityDouble(beta, modelML.globalB)
    assert(D_==(sg2 * (n - c) / n, modelML.globalS2))

    // Now testing association per variant
    // First solve directly with Cholesky
    val directResult = (0 until mG).map { j =>
      val x = convert(G(::, j to j), Double)
      val xC = DenseMatrix.horzcat(x, C)
      val xCc = invChol * xC
      val beta = (xCc.t * xCc) \ (xCc.t * yc)
      val res = norm(yc - xCc * beta)
      val sg2 = (res * res) / (n - c)
      (Variant("1", j + 1, "A", "C"), (beta(0), sg2))
    }.toMap

    // Then solve with LinearMixedModel and compare

    val pheno = y.toArray
    val covSA = (1 until c).map(i => s"sa.covs.cov$i").toArray
    val covSchema = TStruct((1 until c).map(i => (s"cov$i", TDouble)): _*)
    val covData = bnm.sampleIds.zipWithIndex.map { case (id, i) => (id, Annotation.fromSeq( C(i, 1 until c).t.toArray)) }.toMap

    val assocVds = bnm
      .annotateSamples(bnm.sampleIds.zip(pheno).toMap, TDouble, "sa.pheno")
      .annotateSamples(covData, covSchema, "sa.covs")
    val kinshipVds = assocVds.filterVariants((v, va, gs) => v.start <= mW)

    val vds = LinearMixedRegression(assocVds, kinshipVds, "sa.pheno", covSA = covSA,
      useML = false, rootGA = "global.lmmreg", rootVA = "va.lmmreg", runAssoc = true, optDelta = Some(delta),
      sparsityThreshold = 1.0, forceBlock = false, forceGrammian = false)

    val qBeta = vds.queryVA("va.lmmreg.beta")._2
    val qSg2 = vds.queryVA("va.lmmreg.sigmaG2")._2

    val annotationMap = vds.variantsAndAnnotations.collect().toMap

    def assertDouble(q: Querier, v: Variant, value: Double) = {
      val x = q(annotationMap(v)).get.asInstanceOf[Double]
      // println(x, value, v)
      assert(D_==(x, value))
    }

    (0 until mG).foreach { j =>
      val v = Variant("1", j + 1, "A", "C")
      val (beta, sg2) = directResult(v)
      assertDouble(qBeta, v, beta)
      assertDouble(qSg2, v, sg2)
    }
  }

  // Fix parameters, generate y, fit parameters, compare
  // To try with different parameters, remove asserts and add back print statements at the end
  @Test def genAndFitLMM() {
    val seed = 0
    scala.util.Random.setSeed(seed)

    val n = 500
    val c = 2
    val m0 = 1000
    val k = 3
    val Fst = .5

    val FstOfPop = Array.fill[Double](k)(Fst)

    val bnm = BaldingNicholsModel(sc, k, n, m0, None, Some(FstOfPop), scala.util.Random.nextInt(), None, UniformDist(.1, .9))

    val G = TestUtils.removeConstantCols(TestUtils.vdsToMatrixInt(bnm))

    val mG = G.cols
    val mW = mG

    // println(s"$mG of $m0 variants are not constant")

    val W = convert(G(::, 0 until mW), Double)

    // each row has mean 0, norm sqrt(n), variance 1
    for (i <- 0 until mW) {
      W(::, i) -= mean(W(::, i))
      W(::, i) *= math.sqrt(n) / norm(W(::, i))
    }

    val rrm = (W * W.t) / mW.toDouble

    val delta = scala.util.Random.nextDouble()
    val sigmaG2 = scala.util.Random.nextDouble() + 0.5
    val sigmaE2 = delta * sigmaG2

    // Now testing global model
    // First solve directly with Cholesky
    val V = sigmaG2 * rrm + sigmaE2 * DenseMatrix.eye[Double](n)

    val chol = cholesky(V)
    val z = DenseVector.fill[Double](n)(scala.util.Random.nextGaussian())

    val C =
      if (c == 1)
        DenseMatrix.ones[Double](n, 1)
      else
        DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(scala.util.Random.nextGaussian()))

    val beta = DenseVector.fill[Double](c)(scala.util.Random.nextGaussian())

    val y = C * beta + chol * z

    val pheno = y.toArray
    val covSA = (1 until c).map(i => s"sa.covs.cov$i").toArray
    val covSchema = TStruct((1 until c).map(i => (s"cov$i", TDouble)): _*)
    val covData = bnm.sampleIds.zipWithIndex.map { case (id, i) => (id, Annotation.fromSeq( C(i, 1 until c).t.toArray)) }.toMap

    val assocVds = bnm
      .annotateSamples(bnm.sampleIds.zip(pheno).toMap, TDouble, "sa.pheno")
      .annotateSamples(covData, covSchema, "sa.covs")

    val kinshipVds = assocVds.filterVariants((v, va, gs) => v.start <= mW)

    val vds = LinearMixedRegression(assocVds, kinshipVds, "sa.pheno", covSA = covSA,
      useML = false, rootGA = "global.lmmreg", rootVA = "va.lmmreg", runAssoc = true, optDelta = None,
      sparsityThreshold = 1.0, forceBlock = false, forceGrammian = false)

    // val sb = new StringBuilder()
    // sb.append("Global annotation schema:\n")
    // sb.append("global: ")
    // vds.metadata.globalSignature.pretty(sb, 0, printAttrs = true)
    // println(sb.result())

    val fitDelta = vds.queryGlobal("global.lmmreg.delta")._2.get.asInstanceOf[Double]
    val fitSigmaG2 = vds.queryGlobal("global.lmmreg.sigmaG2")._2.get.asInstanceOf[Double]
    val fitBeta = vds.queryGlobal("global.lmmreg.beta")._2.get.asInstanceOf[Map[String, Double]]

    //Making sure type on this is not an array, but an IndexedSeq.
    val evals = vds.queryGlobal("global.lmmreg.evals")._2.get.asInstanceOf[IndexedSeq[Double]]

    val linBeta = (C.t * C) \ (C.t * y)
    val linRes = norm(y - C * linBeta)
    val linSigma2 = (linRes * linRes) / (n - c)

    // println(s"truth / lmm: delta   = $delta / $fitDelta")
    // println(s"truth / lmm / lin: sigmaG2 = $sigmaG2 / $fitSigmaG2 / $linSigma2")
    // println(s"truth / lmm / lin: beta(0) = ${beta(0)} / ${fitBeta("intercept")} / ${linBeta(0)}")
    // (1 until c).foreach( i => println(s"truth / lmm / lin: beta($i) = ${beta(i)} / ${fitBeta(s"sa.covs.cov$i")} / ${linBeta(i)}"))

    assert(D_==(delta, 0.8314409887870612))
    assert(D_==(fitDelta, 0.8410147169942509))
    assert(D_==(delta, fitDelta, 0.05))
    assert(D_==(sigmaG2, fitSigmaG2, 0.05))
    assert(math.abs(beta(0) - fitBeta("intercept")) < 0.05)
    assert(math.abs(beta(1) - fitBeta("sa.covs.cov1")) < 0.05)
  }
}