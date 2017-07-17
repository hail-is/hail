package is.hail.methods

import breeze.linalg._
import breeze.numerics.{sigmoid, sqrt}
import is.hail.annotations._
import is.hail.expr._
import is.hail.stats._
import is.hail.stats.eigSymD.DenseEigSymD
import is.hail.utils._
import is.hail.variant.VariantDataset

import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.univariate.{BrentOptimizer, SearchInterval, UnivariateObjectiveFunction}
import org.apache.commons.math3.util.FastMath

object LinearMixedRegression {
  val schema: Type = TStruct(
    ("beta", TDouble),
    ("sigmaG2", TDouble),
    ("chi2", TDouble),
    ("pval", TDouble))

  def apply(
    assocVds: VariantDataset,
    kinshipMatrix: KinshipMatrix,
    yExpr: String,
    covExpr: Array[String],
    useML: Boolean,
    rootGA: String,
    rootVA: String,
    runAssoc: Boolean,
    optDelta: Option[Double],
    sparsityThreshold: Double,
    useDosages: Boolean,
    optNEigs: Option[Int],
    optDroppedVarianceFraction: Option[Double]): VariantDataset = {

    require(assocVds.wasSplit)

    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(assocVds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = assocVds.sampleIds.map(completeSamplesSet).toArray
    val completeSampleIndex = (0 until assocVds.nSamples)
      .filter(i => completeSamplesSet(assocVds.sampleIds(i)))
      .toArray

    optDelta.foreach(delta =>
      if (delta <= 0d)
        fatal(s"delta must be positive, got ${ delta }"))

    val covNames = "intercept" +: covExpr

    val filteredKinshipMatrix = if (kinshipMatrix.sampleIds sameElements completeSamples)
      kinshipMatrix
    else {
      val fkm = kinshipMatrix.filterSamples(completeSamplesSet)
      if (!(fkm.sampleIds sameElements completeSamples))
        fatal("Array of sample IDs in assoc_vds and array of sample IDs in kinship_matrix (with both filtered to complete " +
          "samples in assoc_vds) do not agree. This should not happen when kinship_matrix is computed from a filtered version of assoc_vds.")
      fkm
    }

    val n = y.length
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"lmmreg: running lmmreg on $n samples with $k sample ${ plural(k, "covariate") } including intercept...")

    val K = new DenseMatrix[Double](n, n, filteredKinshipMatrix.matrix.toBlockMatrixDense().toLocalMatrix().toArray)

    info(s"lmmreg: Computing eigendecomposition of kinship matrix...")

    val eigK = eigSymD(K)
    val fullU = eigK.eigenvectors
    val fullS = eigK.eigenvalues // increasing order

    optDelta match {
      case Some(_) => info(s"lmmreg: Delta specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    val nEigs = (optNEigs, optDroppedVarianceFraction) match {
      case (Some(e), Some(dvf)) => e min computeNEigsDVF(fullS, dvf)
      case (Some(e), None) => e
      case (None, Some(dvf)) => computeNEigsDVF(fullS, dvf)
      case (None, None) => n
    }

    require(nEigs > 0 && nEigs <= n, s"lmmreg: number of kinship eigenvectors to use must be between 1 and the number of samples $n inclusive: got $nEigs")

    val Ut = fullU(::, (n - nEigs) until n).t
    val S = fullS((n - nEigs) until n)

    info(s"lmmreg: Evals 1 to ${math.min(20, nEigs)}: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Evals $nEigs to ${math.max(1, nEigs - 20)}: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val lmmConstants = LMMConstants(y, cov, S, Ut)

    val diagLMM = DiagLMM(lmmConstants, optDelta, useML)

    val delta = diagLMM.delta
    val globalBetaMap = covNames.zip(diagLMM.globalB.toArray).toMap
    val globalSg2 = diagLMM.globalS2
    val globalSe2 = delta * globalSg2
    val h2 = 1 / (1 + delta)

    val header = "rank\teval"
    val evalString = (0 until nEigs).map(i => s"$i\t${ S(nEigs - i - 1) }").mkString("\n")
    log.info(s"\nlmmreg: table of eigenvalues\n$header\n$evalString\n")

    info(s"lmmreg: global model fit: beta = $globalBetaMap")
    info(s"lmmreg: global model fit: sigmaG2 = $globalSg2")
    info(s"lmmreg: global model fit: sigmaE2 = $globalSe2")
    info(s"lmmreg: global model fit: delta = $delta")
    info(s"lmmreg: global model fit: h2 = $h2")

    diagLMM.optGlobalFit.foreach { gf =>
      info(s"lmmreg: global model fit: seH2 = ${ gf.sigmaH2 }")
    }

    val vds1 = assocVds.annotateGlobal(
      Annotation(useML, globalBetaMap, globalSg2, globalSe2, delta, h2, fullS.data.reverse: IndexedSeq[Double], nEigs, optDroppedVarianceFraction.getOrElse(null)),
      TStruct(("useML", TBoolean), ("beta", TDict(TString, TDouble)), ("sigmaG2", TDouble), ("sigmaE2", TDouble),
        ("delta", TDouble), ("h2", TDouble), ("evals", TArray(TDouble)), ("nEigs", TInt), ("dropped_variance_fraction", TDouble)), rootGA)

    val vds2 = diagLMM.optGlobalFit match {
      case Some(gf) =>
        val (logDeltaGrid, logLkhdVals) = gf.gridLogLkhd.unzip
        vds1.annotateGlobal(
          Annotation(gf.sigmaH2, gf.h2NormLkhd, gf.maxLogLkhd, logDeltaGrid, logLkhdVals),
          TStruct(("seH2", TDouble), ("normLkhdH2", TArray(TDouble)), ("maxLogLkhd", TDouble),
            ("logDeltaGrid", TArray(TDouble)), ("logLkhdVals", TArray(TDouble))), rootGA + ".fit")
      case None =>
        assert(optDelta.isDefined)
        vds1
    }

    if (runAssoc) {
      val sc = assocVds.sparkContext
      val sampleMaskBc = sc.broadcast(sampleMask)
      val completeSampleIndexBc = sc.broadcast(completeSampleIndex)

      val (newVAS, inserter) = vds2.insertVA(LinearMixedRegression.schema, pathVA)

      info(s"lmmreg: Computing statistics for each variant...")

      val scalerLMM = if (nEigs == n) {
        val T = Ut(::, *) :* diagLMM.sqrtInvD
        val Qt = qr.reduced.justQ(diagLMM.TC).t
        val QtTy = Qt * diagLMM.Ty
        val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)
        new FullRankScalerLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, T, diagLMM.logNullS2, useML)
      }
      else
        new LowRankScalerLMM(lmmConstants, delta, diagLMM.logNullS2, useML)

      val scalerLMMBc = sc.broadcast(scalerLMM)

      vds2.mapAnnotations { case (v, va, gs) =>
        val x: Vector[Double] =
          if (!useDosages) {
            val x0 = RegressionUtils.hardCalls(gs, n, sampleMaskBc.value)
            if (x0.used <= sparsityThreshold * n) x0 else x0.toDenseVector
          } else
            RegressionUtils.dosages(gs, completeSampleIndexBc.value)
        
        // TODO constant checking to be removed in 0.2
        val nonConstant = useDosages || !RegressionUtils.constantVector(x)

        val lmmregAnnot = if (nonConstant) scalerLMMBc.value.likelihoodRatioTest(x) else null

        val newAnnotation = inserter(va, lmmregAnnot)
        assert(newVAS.typeCheck(newAnnotation))
        newAnnotation
      }.copy(vaSignature = newVAS)
    }
    else
      vds2
  }

  def computeNEigsDVF(S: DenseVector[Double], droppedVarianceFraction: Double): Int = {
    require(0 <= droppedVarianceFraction && droppedVarianceFraction < 1)

    val trace = sum(S)
    var i = -1
    var runningSum = 0.0
    val target = droppedVarianceFraction * trace
    while (runningSum <= target && i < S.length - 1) {
      i += 1
      //Note that S is increasing
      runningSum += S(i)
    }
    S.length - i
  }
}

trait ScalerLMM {
  def likelihoodRatioTest(v: Vector[Double]): Annotation
}

// Handles full-rank case
class FullRankScalerLMM(
  y: DenseVector[Double],
  yy: Double,
  Qt: DenseMatrix[Double],
  Qty: DenseVector[Double],
  yQty: Double,
  T: DenseMatrix[Double],
  logNullS2: Double,
  useML: Boolean) extends ScalerLMM {

  val n = y.length
  val invDf = 1.0 / (if (useML) n else n - Qt.rows)

  def likelihoodRatioTest(x0: Vector[Double]): Annotation = {
    val x = T * x0
    val n = y.length
    val Qtx = Qt * x
    val xQtx: Double = (x dot x) - (Qtx dot Qtx)
    val xQty: Double = (x dot y) - (Qtx dot Qty)

    val b: Double = xQty / xQtx
    val s2 = invDf * (yQty - xQty * b)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    Annotation(b, s2, chi2, p)
  }
}

// Handles low-rank case, but is slower than ScalerLMM on full-rank case
class LowRankScalerLMM(con: LMMConstants, delta: Double, logNullS2: Double, useML: Boolean) extends ScalerLMM {
  val n = con.n
  val d = con.d
  val k = con.S.length
  val y = con.y
  val covt = con.C.t
  val Ut = con.Ut
  val Uty = con.Uty
  val Utcov = con.UtC
  val covtcov = con.CtC
  val covty = con.Cty
  
  val invDf = 1d / (if (useML) n else n - d)
  val invDelta = 1 / delta
  
  val Z = (con.S + delta).map(1 / _ - invDelta)
  val ydy = con.yty / delta + (Uty dot (Uty :* Z))
  val UtcovZUtcov = Utcov.t * (Utcov(::, *) :* Z)
  
  val r1 = 0 to 0
  val r2 = 1 to d
  
  def likelihoodRatioTest(x: Vector[Double]): Annotation = {
    
    val CtC = DenseMatrix.zeros[Double](d + 1, d + 1)
    CtC(0, 0) = x dot x
    CtC(r1, r2) :=  covt * x
    CtC(r2, r1) := CtC(r1, r2).t
    CtC(r2, r2) := con.CtC

    val Utx = Ut * x
    val UtC = DenseMatrix.horzcat(Utx.toDenseMatrix.t, Utcov)
    val ZUtx = Utx :* Z
    
    val Cty = DenseVector.vertcat(DenseVector(x dot y), covty)
    val Cdy = invDelta * Cty + (UtC.t  * (Uty :* Z))

    val CzC = DenseMatrix.zeros[Double](d + 1, d + 1)
    CzC(0, 0) = Utx dot ZUtx
    CzC(r1, r2) := Utcov.t * ZUtx
    CzC(r2, r1) := CzC(r1, r2).t
    CzC(r2, r2) := UtcovZUtcov

    val CdC = invDelta * CtC + CzC
    
    val b = CdC \ Cdy
    val s2 = invDf * (ydy - (Cdy dot b))
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    Annotation(b(0), s2, chi2, p)
  }
}


object DiagLMM {
  def apply(
    lmmConstants: LMMConstants,
    optDelta: Option[Double] = None,
    useML: Boolean = false): DiagLMM = {

    val UtC = lmmConstants.UtC
    val Uty = lmmConstants.Uty

    val CtC = lmmConstants.CtC
    val Cty = lmmConstants.Cty
    val yty = lmmConstants.yty
    val S = lmmConstants.S

    val n = lmmConstants.n
    val d = lmmConstants.d

    def fitDelta(): (Double, GlobalFitLMM) = {

      object LogLkhdML extends UnivariateFunction {
        val shift = -0.5 * n * (1 + math.log(2 * math.Pi))

        def value(logDelta: Double): Double = {
          val delta = FastMath.exp(logDelta)
          val invDelta = 1 / delta
          val D = S + delta
          val dy = Uty :/ D
          val Z = D.map(1 / _ - invDelta)

          val ydy = invDelta * yty + (Uty dot (Uty :* Z))
          val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))
          val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) :* Z))

          val b = CdC \ Cdy
          val sigma2 = (ydy - (Cdy dot b)) / n

          val logdetD = sum(breeze.numerics.log(D)) + (n - S.length) * logDelta

          -0.5 * (logdetD + n * math.log(sigma2)) + shift
        }
      }

      object LogLkhdREML extends UnivariateFunction {
        val shift = -0.5 * (n - d) * (1 + math.log(2 * math.Pi))

        def value(logDelta: Double): Double = {
          val delta = FastMath.exp(logDelta)
          val invDelta = 1 / delta
          val D = S + delta
          val dy = Uty :/ D
          val Z = D.map(1 / _ - invDelta)

          val ydy = invDelta * yty + (Uty dot (Uty :* Z))
          val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))
          val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) :* Z))

          val b = CdC \ Cdy
          val sigma2 = (ydy - (Cdy dot b)) / (n - d)

          val logdetD = sum(breeze.numerics.log(D)) + (n - S.length) * logDelta
          val (_, logdetCdC) = logdet(CdC)
          val (_, logdetCtC) = logdet(CtC)

          -0.5 * (logdetD + logdetCdC - logdetCtC + (n - d) * math.log(sigma2)) + shift
        }
      }

      // number of points per unit of log space
      val pointsPerUnit = 100
      val minLogDelta = -8
      val maxLogDelta = 8

      // avoids rounding of (minLogDelta to logMax by logres)
      val grid = (minLogDelta * pointsPerUnit to maxLogDelta * pointsPerUnit).map(_.toDouble / pointsPerUnit)
      val logLkhdFunction = if (useML) LogLkhdML else LogLkhdREML

      val gridLogLkhd = grid.map(logDelta => (logDelta, logLkhdFunction.value(logDelta)))

      val header = "logDelta\tlogLkhd"
      val gridValsString = gridLogLkhd.map { case (d, nll) => s"$d\t$nll" }.mkString("\n")
      log.info(s"\nlmmreg: table of delta\n$header\n$gridValsString\n")

      val (approxLogDelta, _) = gridLogLkhd.maxBy(_._2)

      if (approxLogDelta == minLogDelta)
        fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta lower search boundary e^$minLogDelta = ${FastMath.exp(minLogDelta)}, indicating negligible enviromental component of variance. The model is likely ill-specified.")
      else if (approxLogDelta == maxLogDelta)
        fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta upper search boundary e^$maxLogDelta = ${FastMath.exp(maxLogDelta)}, indicating negligible genetic component of variance. Standard linear regression may be more appropriate.")

      val searchInterval = new SearchInterval(minLogDelta, maxLogDelta, approxLogDelta)
      val goal = GoalType.MAXIMIZE
      val objectiveFunction = new UnivariateObjectiveFunction(logLkhdFunction)
      // tol = 5e-8 * abs((ln(delta))) + 5e-7 <= 1e-6
      val brentOptimizer = new BrentOptimizer(5e-8, 5e-7)
      val logDeltaPointValuePair = brentOptimizer.optimize(objectiveFunction, goal, searchInterval, MaxEval.unlimited)

      val maxlogDelta = logDeltaPointValuePair.getPoint
      val maxLogLkhd = logDeltaPointValuePair.getValue

      if (math.abs(maxlogDelta - approxLogDelta) > 1d / pointsPerUnit) {
        warn(s"lmmreg: the difference between the optimal value $approxLogDelta of ln(delta) on the grid and" +
          s"the optimal value $maxlogDelta of ln(delta) by Brent's method exceeds the grid resolution" +
          s"of ${1d / pointsPerUnit}. Plot the values over the full grid to investigate.")
      }

      val epsilon = 1d / pointsPerUnit

      // three values of ln(delta) right of, at, and left of the MLE
      val z1 = maxlogDelta + epsilon
      val z2 = maxlogDelta
      val z3 = maxlogDelta - epsilon

      // three values of h2 = sigmoid(-ln(delta)) left of, at, and right of the MLE
      val x1 = sigmoid(-z1)
      val x2 = sigmoid(-z2)
      val x3 = sigmoid(-z3)

      // corresponding values of logLkhd
      val y1 = logLkhdFunction.value(z1)
      val y2 = maxLogLkhd
      val y3 = logLkhdFunction.value(z3)

      if (y1 >= y2 || y3 >= y2)
        fatal(s"Maximum likelihood estimate ${math.exp(maxlogDelta)} for delta is not a global max. " +
          s"Plot the values over the full grid to investigate.")

      // fitting parabola logLkhd ~ a * x^2 + b * x + c near MLE by Lagrange interpolation gives
      // a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x2 - x1) * (x1 - x3) * (x3 - x2))
      // comparing to normal approx: logLkhd ~ 1 / (-2 * sigma^2) * x^2 + lower order terms:
      val sigmaH2 =
      math.sqrt(((x2 - x1) * (x1 - x3) * (x3 - x2)) / (-2 * (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2))))

      val h2LogLkhd = (0.01 to 0.99 by 0.01).map(h2 => logLkhdFunction.value(math.log((1 - h2) / h2)))

      val h2Lkhd = h2LogLkhd.map(ll => math.exp(ll - maxLogLkhd))
      val h2LkhdSum = h2Lkhd.sum
      val h2NormLkhd = IndexedSeq(Double.NaN) ++ h2Lkhd.map(_ / h2LkhdSum) ++ IndexedSeq(Double.NaN)

      (FastMath.exp(maxlogDelta), GlobalFitLMM(maxLogLkhd, gridLogLkhd, sigmaH2, h2NormLkhd))
    }

    def fitUsingDelta(delta: Double, optGlobalFit: Option[GlobalFitLMM]): DiagLMM = {
      val invDelta = 1 / delta
      val invD = (S + delta).map(1 / _)
      val dy = Uty :* invD

      val Z = invD - invDelta

      val ydy = invDelta * yty + (Uty dot (Uty :* Z))
      val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))
      val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) :* Z))

      val b = CdC \ Cdy
      val s2 = (ydy - (Cdy dot b)) / (if (useML) n else n - d)
      val sqrtInvD = sqrt(invD)
      val TC = UtC(::, *) :* sqrtInvD
      val Ty = Uty :* sqrtInvD
      val TyTy = Ty dot Ty

      DiagLMM(b, s2, math.log(s2), delta, optGlobalFit, sqrtInvD, TC, Ty, TyTy, useML)
    }

    val (delta, optGlobalFit) = optDelta match {
      case Some(delta0) => (delta0, None)
      case None => {
        val (delta0, gf) = fitDelta()
        (delta0, Some(gf))
      }
    }

    fitUsingDelta(delta, optGlobalFit)
  }
}

case class GlobalFitLMM(maxLogLkhd: Double, gridLogLkhd: IndexedSeq[(Double, Double)], sigmaH2: Double, h2NormLkhd: IndexedSeq[Double])

case class DiagLMM(
  globalB: DenseVector[Double],
  globalS2: Double,
  logNullS2: Double,
  delta: Double,
  optGlobalFit: Option[GlobalFitLMM],
  sqrtInvD: DenseVector[Double],
  TC: DenseMatrix[Double],
  Ty: DenseVector[Double],
  TyTy: Double,
  useML: Boolean)

object LMMConstants {
  def apply(y: DenseVector[Double], C: DenseMatrix[Double],
            S: DenseVector[Double], Ut: DenseMatrix[Double]): LMMConstants = {
    val UtC = Ut * C
    val Uty = Ut * y

    val CtC = C.t * C
    val Cty = C.t * y
    val yty = y.t * y

    val n = y.length
    val d = C.cols

    new LMMConstants(y, C, S, Ut, Uty, UtC, Cty, CtC, yty, n, d)
  }
}

case class LMMConstants(y: DenseVector[Double], C: DenseMatrix[Double], S: DenseVector[Double], Ut: DenseMatrix[Double],
                        Uty: DenseVector[Double], UtC: DenseMatrix[Double], Cty: DenseVector[Double],
                        CtC: DenseMatrix[Double], yty: Double, n: Int, d: Int)
