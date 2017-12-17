package is.hail.methods

import breeze.linalg._
import breeze.numerics.{abs, exp, sigmoid}
import breeze.stats.mean
import is.hail.annotations._
import is.hail.expr.{TArray, TFloat64, TStruct}
import is.hail.stats._
import is.hail.utils._
import is.hail.variant.{Variant, MatrixTable}
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class LinearMixedRegressionSuite extends SparkSuite {

  def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6) {
    assert(D_==(a.asInstanceOf[Double], value, tol))
  }
  
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
    V.forceSymmetry()
    
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

    val constants = LMMConstants(y, C, S, Ut)

    val model = DiagLMM(LMMConstants(y, C, S, Ut), optDelta = Some(delta))

    TestUtils.assertVectorEqualityDouble(beta, model.globalB)
    assert(D_==(sg2, model.globalS2))

    val modelML = DiagLMM(LMMConstants(y, C, S, Ut), optDelta = Some(delta), useML = true)

    TestUtils.assertVectorEqualityDouble(beta, modelML.globalB)
    assert(D_==(sg2 * (n - c) / n, modelML.globalS2))
    
    def lmmfit(x: DenseMatrix[Double]): (Double, Double, Double, Double) = {
      val xC = DenseMatrix.horzcat(x, C)
      val yc = invChol * y
      val xCc = invChol * xC
      val beta = (xCc.t * xCc) \ (xCc.t * yc)
      val res = norm(yc - xCc * beta)
      val sg2 = (res * res) / (n - c)
      val chi2 = n * (model.logNullS2 - math.log(sg2))
      val pval = chiSquaredTail(1d, chi2)
      
      (beta(0), sg2, chi2, pval)
    }

    // Now testing association per variant
    // Solve directly with Cholesky
    val directResult = (0 until mG).map { j => 
      (Variant("1", j + 1, "A", "C"), lmmfit(convert(G(::, j to j), Double))) }.toMap

    // Then solve with LinearMixedModel and compare
    val vds0 = vdsFromGtMatrix(hc)(G)
    val pheno = y.toArray
    val cov1 = C(::, 1).toArray
    val cov2 = C(::, 2).toArray

    val assocVDS = vds0
      .annotateSamples(vds0.sampleIds.zip(pheno).toMap, TFloat64(), "sa.pheno")
      .annotateSamples(vds0.sampleIds.zip(cov1).toMap, TFloat64(), "sa.cov1")
      .annotateSamples(vds0.sampleIds.zip(cov2).toMap, TFloat64(), "sa.cov2")
    val kinshipVDS = assocVDS.filterVariants((v, va, gs) => v.asInstanceOf[Variant].start <= 2)

    val vds = assocVDS.lmmreg(ComputeRRM(kinshipVDS), "sa.pheno", "g.GT.nNonRefAlleles()", covariates = Array("sa.cov1", "sa.cov2"), delta = Some(delta))

    val qBeta = vds.queryVA("va.lmmreg.beta")._2
    val qSg2 = vds.queryVA("va.lmmreg.sigmaG2")._2
    val qChi2 = vds.queryVA("va.lmmreg.chi2")._2
    val qPval = vds.queryVA("va.lmmreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    (0 until mG).foreach { j =>
      val v = Variant("1", j + 1, "A", "C")
      val (beta, sg2, chi2, pval) = directResult(v)
      assertDouble(qBeta(a(v)), beta)
      assertDouble(qSg2(a(v)), sg2)
      assertDouble(qChi2(a(v)), chi2)
      assertDouble(qPval(a(v)), pval)
    }
    
    // test dosages
    val gpMat = DenseMatrix(
      (Array(0.1, 0.2, 0.7), null),
      (Array(0.0, 0.2, 0.8), Array(1.0, 0.0, 0.0)),
      (Array(0.5, 0.2, 0.3), Array(0.4, 0.3, 0.3)),
      (Array(0.6, 0.2, 0.2), Array(0.2, 0.2, 0.6)),
      (Array(1.0, 0.0, 0.0), Array(0.1, 0.2, 0.7)),
      (Array(0.0, 1.0, 0.0), Array(0.9, 0.1, 0.0)))
    
    val dosageMat = gpMat.map(a => if (a != null) a(1) + 2 * a(2) else 0)
    dosageMat.update(0, 1, (sum(dosageMat(::, 1)) - dosageMat(0, 1)) / 5) // mean impute missing value
    
    val vds1 = vdsFromGpMatrix(hc)(nAlleles = 2, gpMat)
      .annotateSamples(vds0.sampleIds.zip(pheno).toMap, TFloat64(), "sa.pheno")
      .annotateSamples(vds0.sampleIds.zip(cov1).toMap, TFloat64(), "sa.cov1")
      .annotateSamples(vds0.sampleIds.zip(cov2).toMap, TFloat64(), "sa.cov2")
      .lmmreg(ComputeRRM(kinshipVDS), "sa.pheno", "dosage(g.GP)", covariates = Array("sa.cov1", "sa.cov2"), delta = Some(delta), optDroppedVarianceFraction = Some(0))
    
    val directResult1 = (0 until 2).map { j => (Variant("1", j + 1, "A", "C"), lmmfit(dosageMat(::, j to j))) }.toMap
    
    val qBeta1 = vds1.queryVA("va.lmmreg.beta")._2
    val qSg21 = vds1.queryVA("va.lmmreg.sigmaG2")._2
    val qChi21 = vds1.queryVA("va.lmmreg.chi2")._2
    val qPval1 = vds1.queryVA("va.lmmreg.pval")._2

    val a1 = vds1.variantsAndAnnotations.collect().toMap
    
    (0 until 2).foreach { j =>
      val v = Variant("1", j + 1, "A", "C")
      val (beta, sg2, chi2, pval) = directResult1(v)
      assertDouble(qBeta1(a1(v)), beta, 1e-3)
      println(qSg21(a1(v)), sg2)
      assertDouble(qSg21(a1(v)), sg2, 1e-3)
      assertDouble(qChi21(a1(v)), chi2, 1e-3)
      assertDouble(qPval1(a1(v)), pval, 1e-3)
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
        DenseMatrix.horzcat(
          DenseMatrix.ones[Double](n, 1),
          DenseMatrix.fill[Double](n, c - 1)(scala.util.Random.nextGaussian()))

    val FstOfPop = Array.fill[Double](k)(Fst)

    val bnm = BaldingNicholsModel(hc, k, n, m0, None, Some(FstOfPop),
      scala.util.Random.nextInt(), Some(4), UniformDist(.1, .9))

    val G = TestUtils.removeConstantCols(TestUtils.vdsToMatrixInt(bnm))

    val mG = G.cols
    val mW = G.cols

    // println(s"$mG of $m0 variants are not constant")

    val W = convert(G(::, 0 until mW), Double)

    // each row has mean 0, norm sqrt(n), variance 1
    for (i <- 0 until mW) {
      W(::, i) -= mean(W(::, i))
      W(::, i) *= math.sqrt(n) / norm(W(::, i))
    }

    val rrm = (W * W.t) / mW.toDouble // RRM
    val delta = math.exp(10 * scala.util.Random.nextDouble() - 5)

    // Now testing global model
    // First solve directly with Cholesky
    val V = rrm + DenseMatrix.eye[Double](n) * delta
    V.forceSymmetry()
    
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

    val constants = LMMConstants(y, C, S, Ut)

    val model = DiagLMM(constants, optDelta = Some(delta))

    TestUtils.assertVectorEqualityDouble(beta, model.globalB)
    assert(D_==(sg2, model.globalS2))

    val modelML = DiagLMM(constants, optDelta = Some(delta), useML = true)

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
    val covExpr = (1 until c).map(i => s"sa.covs.cov$i").toArray
    val covSchema = TStruct((1 until c).map(i => (s"cov$i", TFloat64())): _*)
    val covData = bnm.sampleIds.zipWithIndex.map { case (id, i) =>
      (id, Annotation.fromSeq( C(i, 1 until c).t.toArray)) }.toMap

    val assocVDS = bnm
      .annotateSamples(bnm.sampleIds.zip(pheno).toMap, TFloat64(), "sa.pheno")
      .annotateSamples(covData, covSchema, "sa.covs")
    val kinshipVDS = assocVDS.filterVariants((v, va, gs) => v.asInstanceOf[Variant].start <= mW)

    val vds = assocVDS.lmmreg(ComputeRRM(kinshipVDS), "sa.pheno", "g.GT.nNonRefAlleles()", covariates = covExpr, delta = Some(delta), optDroppedVarianceFraction = Some(0))

    val qBeta = vds.queryVA("va.lmmreg.beta")._2
    val qSg2 = vds.queryVA("va.lmmreg.sigmaG2")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    (0 until mG).foreach { j =>
      val v = Variant("1", j + 1, "A", "C")
      val (beta, sg2) = directResult(v)
      assertDouble(qBeta(a(v)), beta)
      assertDouble(qSg2(a(v)), sg2)
    }
  }

  /*
  FastLMM Test data is from all.bed, all.bim, all.fam, cov.txt, pheno_10_causals.txt:
    https://github.com/MicrosoftGenomics/FaST-LMM/tree/master/tests/datasets/synth

  Data is filtered to chromosome 1,3 and samples 0-124,375-499 (2000 variants and 250 samples)

  Results are computed with single_snp as in:
    https://github.com/MicrosoftGenomics/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb
  */

  lazy val covariates = hc.importTable("src/test/resources/fastlmmCov.txt",
    noHeader = true, impute = true).keyBy("f1")
  lazy val phenotypes = hc.importTable("src/test/resources/fastlmmPheno.txt",
    noHeader = true, impute = true, separator = " ").keyBy("f1")

  lazy val vdsFastLMM = hc.importPlink(bed = "src/test/resources/fastlmmTest.bed",
    bim = "src/test/resources/fastlmmTest.bim",
    fam = "src/test/resources/fastlmmTest.fam")
    .annotateSamplesTable(covariates, expr = "sa.cov=table.f2")
    .annotateSamplesTable(phenotypes, expr = "sa.pheno=table.f2")

  lazy val vdsChr1 = vdsFastLMM.filterVariantsExpr("""v.contig == "1"""")
    .lmmreg(ComputeRRM(vdsFastLMM.filterVariantsExpr("""v.contig != "1"""")), "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov"), runAssoc = false)

  lazy val vdsChr3 = vdsFastLMM.filterVariantsExpr("""v.contig == "3"""")
    .lmmreg(ComputeRRM(vdsFastLMM.filterVariantsExpr("""v.contig != "3"""")), "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov"), runAssoc = false)

  @Test def fastLMMTest() {
    val h2Chr1 = vdsChr1.queryGlobal("global.lmmreg.h2")._2.asInstanceOf[Double]
    val h2Chr3 = vdsChr3.queryGlobal("global.lmmreg.h2")._2.asInstanceOf[Double]

    assert(D_==(h2Chr1, 0.36733240))
    assert(D_==(h2Chr3, 0.14276117))
  }

  @Test def h2seTest() {
    // Testing that the parabolic approximation of h2 standard error is close to the empirical standard deviation of the
    // normalized likelihood function, e.g. the posterior with uniform prior on [0,1].

    val h2Chr1 = vdsChr1.queryGlobal("global.lmmreg.h2")._2.asInstanceOf[Double]
    val h2Chr3 = vdsChr3.queryGlobal("global.lmmreg.h2")._2.asInstanceOf[Double]

    val seH2Chr1 = vdsChr1.queryGlobal("global.lmmreg.fit.seH2")._2.asInstanceOf[Double]
    val seH2Chr3 = vdsChr3.queryGlobal("global.lmmreg.fit.seH2")._2.asInstanceOf[Double]

    val logDeltaGrid =
      DenseVector(vdsChr1.queryGlobal("global.lmmreg.fit.logDeltaGrid")._2.asInstanceOf[IndexedSeq[Double]].toArray)

    val logLkhdVals1 =
      DenseVector(vdsChr1.queryGlobal("global.lmmreg.fit.logLkhdVals")._2.asInstanceOf[IndexedSeq[Double]].toArray)
    val logLkhdVals3 =
      DenseVector(vdsChr3.queryGlobal("global.lmmreg.fit.logLkhdVals")._2.asInstanceOf[IndexedSeq[Double]].toArray)

    // Construct normalized likelihood function of h2
    // shift log lkhd to have max of 0, to prevent numerical issues
    logLkhdVals1 :-= max(logLkhdVals1)
    logLkhdVals3 :-= max(logLkhdVals3)

    // integrate in h2 coordinates
    val h2Vals = sigmoid(-logDeltaGrid)

    // d(h2) / d (ln(delta)) = - h2 * (1 - h2)
    val widths = h2Vals :* (1d - h2Vals)

    // normalization constant
    val total1 = exp(logLkhdVals1) dot widths
    val total3 = exp(logLkhdVals3) dot widths

    // normalized likelihood of h2
    val h2Posterior1 = (exp(logLkhdVals1) :* widths) :/ total1
    val h2Posterior3 = (exp(logLkhdVals3) :* widths) :/ total3

    // normal approximation to mean and standard deviation
    val meanPosterior1 = sum(h2Vals :* h2Posterior1)
    val meanPosterior3 = sum(h2Vals :* h2Posterior3)

    val sdPosterior1 = math.sqrt(sum((h2Vals :- meanPosterior1) :* (h2Vals :- meanPosterior1) :* h2Posterior1 ))
    val sdPosterior3 = math.sqrt(sum((h2Vals :- meanPosterior3) :* (h2Vals :- meanPosterior3) :* h2Posterior3 ))

    assert(math.abs(h2Chr1 - meanPosterior1) < 0.01) // both are approx 0.37
    assert(math.abs(seH2Chr3 - meanPosterior3) < 0.07) // values are approx 0.14 and 0.20

    assert(math.abs(seH2Chr1 - sdPosterior1) < 0.02) // both are approx 0.16
    assert(math.abs(seH2Chr3 - sdPosterior3) < 0.02) // both are approx 0.13

    val h2NormLkhd1 = DenseVector(
      vdsChr1.queryGlobal("global.lmmreg.fit.normLkhdH2")._2.asInstanceOf[IndexedSeq[Double]].slice(1,100).toArray)

    val h2NormLkhd3 = DenseVector(
      vdsChr3.queryGlobal("global.lmmreg.fit.normLkhdH2")._2.asInstanceOf[IndexedSeq[Double]].slice(1,100).toArray)

    // checking that normLkhdH2 is normalized
    assert(D_==(sum(h2NormLkhd1), 1d))
    assert(D_==(sum(h2NormLkhd3), 1d))

    // comparing normLkhdH2 and approximation of h2Posterior over h2Grid
    val h2Grid = DenseVector((0.01 to 0.99 by 0.01).toArray)

    val h2NormLkhdMean1 = sum(h2Grid :* h2NormLkhd1)
    val h2NormLkhdMean3 = sum(h2Grid :* h2NormLkhd3)

    val h2NormLkhdSe1 = sum((h2Grid - h2NormLkhdMean1) :* (h2Grid - h2NormLkhdMean1) :* h2NormLkhd1)
    val h2NormLkhdSe3 = sum((h2Grid - h2NormLkhdMean3) :* (h2Grid - h2NormLkhdMean3) :* h2NormLkhd3)

    val minLogDelta = -8
    val pointsPerUnit = 100

    def h2ToNearbyIndex(h2: Double): Int = ((math.log((1 - h2) / h2) - minLogDelta) * pointsPerUnit).round.toInt

    val h2Post1 = h2Grid.map(h2 => h2Posterior1(h2ToNearbyIndex(h2)))
    val h2Post3 = h2Grid.map(h2 => h2Posterior3(h2ToNearbyIndex(h2)))

    h2Post1 :/= sum(h2Post1)
    h2Post3 :/= sum(h2Post3)

    val h2PostMean1 = sum(h2Grid :* h2Post1)
    val h2PostMean3 = sum(h2Grid :* h2Post3)

    val h2PostSe1 = math.sqrt(sum((h2Grid - h2PostMean1) :* (h2Grid - h2PostMean1) :* h2Post1))
    val h2PostSe3 = math.sqrt(sum((h2Grid - h2PostMean3) :* (h2Grid - h2PostMean3) :* h2Post3))

    assert(math.abs(meanPosterior1 - h2PostMean1) < 0.03) // approx 0.37 and 0.40
    assert(math.abs(meanPosterior3 - h2PostMean3) < 0.06) // approx 0.20 and 0.26

    assert(math.abs(sdPosterior1 - h2PostSe1) < 0.02) // both are approx 0.15
    assert(math.abs(sdPosterior3 - h2PostSe3) < 0.01) // both are approx 0.12

    assert(max(abs(h2NormLkhd1 - h2Post1)) < .02)
    assert(max(abs(h2NormLkhd3 - h2Post3)) < .02)
  }

  // this test parallels the lmmreg Python test, and is a regression test related to filtering samples first
  @Test def filterTest() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    var vdsAssoc = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .filterMulti()
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno.Pheno")
      .annotateSamplesExpr("""sa.culprit = gs.filter(g => v == Variant("1", 1, "C", "T")).map(g => g.GT.gt).collect()[0]""")
      .annotateSamplesExpr("sa.pheno.PhenoLMM = (1 + 0.1 * sa.cov.Cov1 * sa.cov.Cov2) * sa.culprit")

    val vdsKinship = vdsAssoc.filterVariantsExpr("v.start < 4")

    vdsAssoc = vdsAssoc.lmmreg(ComputeRRM(vdsKinship), "sa.pheno.PhenoLMM", "g.GT.nNonRefAlleles()",
      Array("sa.cov.Cov1", "sa.cov.Cov2"), runAssoc = false)

    vdsAssoc.count()
  }

  //Tests that k eigenvectors give the same result as all n eigenvectors for a rank-k kinship matrix on n samples.
  @Test def testFullRankAndLowRank() {
    val vdsChr1: MatrixTable = vdsFastLMM.filterVariantsExpr("""v.contig == "1"""")

    val notChr1VDSDownsampled = vdsFastLMM.filterVariantsExpr("""v.contig == "3" && v.start < 2242""")

    val rrm = ComputeRRM(notChr1VDSDownsampled)

    //REML TESTS
    val vdsChr1FullRankREML = vdsChr1.lmmreg(rrm, "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov"), runAssoc = false, delta = None)

    val vdsChr1LowRankREML = vdsChr1.lmmreg(rrm, "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov"), runAssoc = false, nEigs = Some(242))

    globalLMMCompare(vdsChr1FullRankREML, vdsChr1LowRankREML)

    //ML TESTS
    val vdsChr1FullRankML = vdsChr1.lmmreg(rrm, "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov"), useML = true, runAssoc = false, delta = None)

    val vdsChr1LowRankML = vdsChr1.lmmreg(rrm, "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov"), useML = true, runAssoc = false, nEigs = Some(242))

    globalLMMCompare(vdsChr1FullRankML, vdsChr1LowRankML)
  }

  private def globalLMMCompare(vds1: MatrixTable, vds2: MatrixTable) {
    assert(D_==(vds1.queryGlobal("global.lmmreg.beta")._2.asInstanceOf[Map[String, Double]].apply("intercept"),
      vds2.queryGlobal("global.lmmreg.beta")._2.asInstanceOf[Map[String, Double]].apply("intercept")))

    assert(D_==(vds1.queryGlobal("global.lmmreg.delta")._2.asInstanceOf[Double],
      vds2.queryGlobal("global.lmmreg.delta")._2.asInstanceOf[Double]))

    assert(D_==(vds1.queryGlobal("global.lmmreg.h2")._2.asInstanceOf[Double],
      vds2.queryGlobal("global.lmmreg.h2")._2.asInstanceOf[Double]))
  }

  lazy val smallMat = DenseMatrix(
    (1, 2, 0),
    (2, 1, 1),
    (1, 1, 1),
    (0, 0, 2),
    (1, 0, 1),
    (0, 1, 1),
    (2, 2, 2),
    (2, 0, 1),
    (1, 0, 0),
    (1, 1, 2))

  val rand = new scala.util.Random()
  rand.setSeed(5)
  val randomNorms = (1 to 10).map(x => rand.nextGaussian())

  lazy val vdsSmall = vdsFromGtMatrix(hc)(smallMat)
    .annotateSamplesExpr("sa.culprit = gs.filter(g => v.start == 2).map(g => g.GT.gt).collect()[0]")
    .annotateGlobal(randomNorms, TArray(TFloat64()), "global.randNorms")
    .annotateSamplesExpr("sa.pheno = sa.culprit + global.randNorms[s.toInt32()]")

  lazy val vdsSmallRRM = ComputeRRM(vdsSmall)
  
  @Test def testSmall() {
    val vdsLmmreg = vdsSmall.lmmreg(vdsSmallRRM, "sa.pheno", "g.GT.nNonRefAlleles()")

    val vdsLmmregLowRank = vdsSmall.lmmreg(vdsSmallRRM, "sa.pheno", "g.GT.nNonRefAlleles()", nEigs = Some(3))

    globalLMMCompare(vdsLmmreg, vdsLmmregLowRank)

    assert(vdsLmmregLowRank.queryGlobal("global.lmmreg.nEigs")._2.asInstanceOf[Int] == 3)
  }

  @Test def testVarianceFraction() {
    val vdsLmmreg = vdsSmall.lmmreg(vdsSmallRRM, "sa.pheno", "g.GT.nNonRefAlleles()", optDroppedVarianceFraction = Some(0.3))
    assert(vdsLmmreg.queryGlobal("global.lmmreg.nEigs")._2 == 2)
    assert(vdsLmmreg.queryGlobal("global.lmmreg.dropped_variance_fraction")._2 == 0.3)
  }

  @Test def computeNEigsDVF() {
    val eigs = DenseVector(0.0, 1.0, 2.0, 3.0, 4.0)
    assert(LinearMixedRegression.computeNEigsDVF(eigs, 0.1) == 3)
    assert(LinearMixedRegression.computeNEigsDVF(eigs, 0.6) == 1)
    assert(LinearMixedRegression.computeNEigsDVF(eigs, 0.59) == 2)
    assert(LinearMixedRegression.computeNEigsDVF(eigs, 0.2) == 3)
  }
}
