package is.hail.methods

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, _}
import breeze.numerics.{sigmoid, sqrt}
import is.hail.annotations._
import is.hail.expr.types._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.univariate.{BrentOptimizer, SearchInterval, UnivariateObjectiveFunction}
import org.apache.commons.math3.util.FastMath

object LinearMixedRegression {
  val schema: Type = TStruct(
    ("beta", TFloat64()),
    ("sigma_g_squared", TFloat64()),
    ("chi_sq_stat", TFloat64()),
    ("p_value", TFloat64()))

  def apply(
    assocVSM: MatrixTable,
    kinshipMatrix: KinshipMatrix,
    yField: String,
    xField: String,
    covFields: Array[String],
    useML: Boolean,
    rootGA: String,
    rootVA: String,
    runAssoc: Boolean,
    optDelta: Option[Double],
    sparsityThreshold: Double,
    optNEigs: Option[Int],
    optDroppedVarianceFraction: Option[Double]): MatrixTable = {

    val (y, cov, completeColIdx) = RegressionUtils.getPhenoCovCompleteSamples(assocVSM, yField, covFields)
    val completeColIds = completeColIdx.map(assocVSM.stringSampleIds)

    optDelta.foreach(delta =>
      if (delta <= 0d)
        fatal(s"delta must be positive, got ${ delta }"))

    val covNames = "intercept" +: covFields

    val filteredKinshipMatrix = if (kinshipMatrix.sampleIds sameElements completeColIds)
      kinshipMatrix
    else {
      val fkm = kinshipMatrix.filterSamples(completeColIds.toSet)
      if (!(fkm.sampleIds sameElements completeColIds))
        fatal("Array of sample IDs in dataset and array of sample IDs in 'kinship_matrix' (with both filtered to complete " +
          "samples in dataset) do not agree.")
      fkm
    }

    val n = y.length
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x and intercept) implies $d degrees of freedom.")

    info(s"linear_mixed_regression: running on $n samples for response variable y,\n"
       + s"    with input variable x, intercept, and ${ k - 1 } additional ${ plural(k - 1, "covariate") }...")

    val K = filteredKinshipMatrix.matrix.toHailBlockMatrix().toBreezeMatrix()

    info(s"linear_mixed_regression: computing eigendecomposition of kinship matrix...")

    val eigK = eigSymD(K)
    val fullU = eigK.eigenvectors
    val fullS = eigK.eigenvalues // increasing order

    optDelta match {
      case Some(_) => info(s"linear_mixed_regression: delta specified by user")
      case None => info(s"linear_mixed_regression: estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    val nEigs = (optNEigs, optDroppedVarianceFraction) match {
      case (Some(e), Some(dvf)) => e min computeNEigsDVF(fullS, dvf)
      case (Some(e), None) => e
      case (None, Some(dvf)) => computeNEigsDVF(fullS, dvf)
      case (None, None) => n
    }

    require(nEigs > 0 && nEigs <= n, s"linear_mixed_regression: number of kinship eigenvectors to use " +
      s"must be between 1 and the number of samples $n inclusive: got $nEigs")

    val Ut = fullU(::, (n - nEigs) until n).t
    val S = fullS((n - nEigs) until n)

    info(s"linear_mixed_regression: eigenvalues 1 to ${ math.min(20, nEigs) }: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"linear_mixed_regression: eigenvalues $nEigs to ${ math.max(1, nEigs - 20) }: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val lmmConstants = LMMConstants(y, cov, S, Ut)

    val diagLMM = DiagLMM(lmmConstants, optDelta, useML)

    val delta = diagLMM.delta
    val globalBetaMap = covNames.zip(diagLMM.globalB.toArray).toMap
    val globalSg2 = diagLMM.globalS2
    val globalSe2 = delta * globalSg2
    val h2 = 1 / (1 + delta)

    val header = "rank\teval"
    val evalString = (0 until nEigs).map(i => s"$i\t${ S(nEigs - i - 1) }").mkString("\n")
    log.info(s"\nlinear_mixed_regression: table of eigenvalues\n$header\n$evalString\n")

    info(s"linear_mixed_regression: global model fit: beta = $globalBetaMap")
    info(s"linear_mixed_regression: global model fit: sigma_g_squared = $globalSg2")
    info(s"linear_mixed_regression: global model fit: sigma_e_squared = $globalSe2")
    info(s"linear_mixed_regression: global model fit: delta = $delta")
    info(s"linear_mixed_regression: global model fit: h_squared = $h2")

    diagLMM.optGlobalFit.foreach { gf =>
      info(s"linear_mixed_regression: global model fit: standard_error_h_squared = ${ gf.sigmaH2 }")
    }

    val fitType = TStruct(
      "standard_error_h_squared" -> TFloat64(),
      "normalized_likelihood_h_squared" -> TArray(TFloat64()),
      "max_log_likelihood" -> TFloat64(),
      "log_delta_grid" -> TArray(TFloat64()),
      "log_likelihood_values" -> TArray(TFloat64())
    )

    val t = TStruct(
      "use_ml" -> TBoolean(),
      "beta" -> TDict(TString(), TFloat64()),
      "sigma_g_squared" -> TFloat64(),
      "sigma_e_squared" -> TFloat64(),
      "delta" -> TFloat64(),
      "h_squared" -> TFloat64(),
      "eigenvalues" -> TArray(TFloat64()),
      "n_eigenvectors" -> TInt32(),
      "dropped_variance_fraction" -> TFloat64(),
      "fit" -> fitType
    )

    val aFit = diagLMM.optGlobalFit match {
      case Some(gf) =>
        val (logDeltaGrid, logLkhdVals) = gf.gridLogLkhd.unzip
        Annotation(gf.sigmaH2, gf.h2NormLkhd, gf.maxLogLkhd, logDeltaGrid, logLkhdVals)
      case None =>
        assert(optDelta.isDefined)
        Annotation(null, null, null, null, null)
    }

    val a = Annotation(useML, globalBetaMap, globalSg2, globalSe2, delta, h2, fullS.data.reverse: IndexedSeq[Double],
      nEigs, optDroppedVarianceFraction.orNull, aFit)

    val vds = assocVSM.annotateGlobal(a, t, rootGA)

    if (runAssoc) {
      val sc = assocVSM.sparkContext
      val completeColIdxBc = sc.broadcast(completeColIdx)

      info(s"linear_mixed_regression: Computing statistics for each variant...")

      val scalerLMM = if (nEigs == n) {
        val T = Ut(::, *) *:* diagLMM.sqrtInvD
        val Qt = qr.reduced.justQ(diagLMM.TC).t
        val QtTy = Qt * diagLMM.Ty
        val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)
        new FullRankScalerLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, T, diagLMM.logNullS2, useML)
      }
      else
        new LowRankScalerLMM(lmmConstants, delta, diagLMM.logNullS2, useML)

      val scalerLMMBc = sc.broadcast(scalerLMM)

      val fullRowType = vds.rvRowType
      val entryArrayType = vds.matrixType.entryArrayType
      val entryType = vds.entryType
      val fieldType = entryType.field(xField).typ

      assert(fieldType.isOfType(TFloat64()))
  
      val entryArrayIdx = vds.entriesIndex
      val fieldIdx = entryType.fieldIdx(xField)
      
      vds.insertIntoRow(() => (new BDV[Double](n), new ArrayBuilder[Int]()))(
        LinearMixedRegression.schema, rootVA, { case ((x, missingCompleteCols), rv, rvb) =>

          RegressionUtils.setMeanImputedDoubles(x.data, 0, completeColIdxBc.value, missingCompleteCols, 
            rv, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)
  
          scalerLMMBc.value.likelihoodRatioTest(x, rvb)
        }
      )
    } else
      vds
  }

  def computeNEigsDVF(S: BDV[Double], droppedVarianceFraction: Double): Int = {
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
  def likelihoodRatioTest(v: Vector[Double], rvb: RegionValueBuilder): Unit
}

// Handles full-rank case
class FullRankScalerLMM(
  y: BDV[Double],
  yy: Double,
  Qt: BDM[Double],
  Qty: BDV[Double],
  yQty: Double,
  T: BDM[Double],
  logNullS2: Double,
  useML: Boolean) extends ScalerLMM {

  val n = y.length
  val invDf = 1.0 / (if (useML) n else n - Qt.rows)

  def likelihoodRatioTest(x0: Vector[Double], rvb: RegionValueBuilder): Unit = {
    val x = T * x0
    val n = y.length
    val Qtx = Qt * x
    val xQtx: Double = (x dot x) - (Qtx dot Qtx)
    val xQty: Double = (x dot y) - (Qtx dot Qty)

    val b: Double = xQty / xQtx
    val s2 = invDf * (yQty - xQty * b)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(chi2, 1)

    rvb.startStruct()
    rvb.addDouble(b)
    rvb.addDouble(s2)
    rvb.addDouble(chi2)
    rvb.addDouble(p)
    rvb.endStruct()
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
  val ydy = con.yty / delta + (Uty dot (Uty *:* Z))
  val UtcovZUtcov = Utcov.t * (Utcov(::, *) *:* Z)

  val r1 = 0 to 0
  val r2 = 1 to d

  def likelihoodRatioTest(x: Vector[Double], rvb: RegionValueBuilder): Unit = {

    val CtC = BDM.zeros[Double](d + 1, d + 1)
    CtC(0, 0) = x dot x
    CtC(r1, r2) := covt * x
    CtC(r2, r1) := CtC(r1, r2).t
    CtC(r2, r2) := con.CtC

    val Utx = Ut * x
    val UtC = BDM.horzcat(Utx.toDenseMatrix.t, Utcov)
    val ZUtx = Utx *:* Z

    val Cty = BDV.vertcat(BDV(x dot y), covty)
    val Cdy = invDelta * Cty + (UtC.t * (Uty *:* Z))

    val CzC = BDM.zeros[Double](d + 1, d + 1)
    CzC(0, 0) = Utx dot ZUtx
    CzC(r1, r2) := Utcov.t * ZUtx
    CzC(r2, r1) := CzC(r1, r2).t
    CzC(r2, r2) := UtcovZUtcov

    val CdC = invDelta * CtC + CzC

    try {
      val b = CdC \ Cdy
      val s2 = invDf * (ydy - (Cdy dot b))
      val chi2 = n * (logNullS2 - math.log(s2))
      val p = chiSquaredTail(chi2, 1)

      rvb.startStruct()
      rvb.addDouble(b(0))
      rvb.addDouble(s2)
      rvb.addDouble(chi2)
      rvb.addDouble(p)
      rvb.endStruct()
    } catch {
      case _: breeze.linalg.MatrixSingularException =>
        rvb.startStruct()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.endStruct()
    }
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
          val Z = D.map(1 / _ - invDelta)

          val ydy = invDelta * yty + (Uty dot (Uty *:* Z))
          val Cdy = invDelta * Cty + (UtC.t * (Uty *:* Z))
          val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) *:* Z))

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
          val Z = D.map(1 / _ - invDelta)

          val ydy = invDelta * yty + (Uty dot (Uty *:* Z))
          val Cdy = invDelta * Cty + (UtC.t * (Uty *:* Z))
          val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) *:* Z))

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

      val header = "log_delta_grid\tlog_likelihood"
      val gridValsString = gridLogLkhd.map { case (d, nll) => s"$d\t$nll" }.mkString("\n")
      log.info(s"\nlinear_mixed_regression: table of delta\n$header\n$gridValsString\n")

      val (approxLogDelta, _) = gridLogLkhd.maxBy(_._2)

      if (approxLogDelta == minLogDelta)
        fatal(s"linear_mixed_regression: failed to fit delta: ${ if (useML) "ML" else "REML" } realized at delta lower search boundary e^$minLogDelta = ${ FastMath.exp(minLogDelta) }, indicating negligible enviromental component of variance. The model is likely ill-specified.")
      else if (approxLogDelta == maxLogDelta)
        fatal(s"linear_mixed_regression: failed to fit delta: ${ if (useML) "ML" else "REML" } realized at delta upper search boundary e^$maxLogDelta = ${ FastMath.exp(maxLogDelta) }, indicating negligible genetic component of variance. Standard linear regression may be more appropriate.")

      val searchInterval = new SearchInterval(minLogDelta, maxLogDelta, approxLogDelta)
      val goal = GoalType.MAXIMIZE
      val objectiveFunction = new UnivariateObjectiveFunction(logLkhdFunction)
      // tol = 5e-8 * abs((ln(delta))) + 5e-7 <= 1e-6
      val brentOptimizer = new BrentOptimizer(5e-8, 5e-7)
      val logDeltaPointValuePair = brentOptimizer.optimize(objectiveFunction, goal, searchInterval, MaxEval.unlimited)

      val maxlogDelta = logDeltaPointValuePair.getPoint
      val maxLogLkhd = logDeltaPointValuePair.getValue

      if (math.abs(maxlogDelta - approxLogDelta) > 1d / pointsPerUnit) {
        warn(s"linear_mixed_regression: the difference between the optimal value $approxLogDelta of ln(delta) on the grid and" +
          s"the optimal value $maxlogDelta of ln(delta) by Brent's method exceeds the grid resolution" +
          s"of ${ 1d / pointsPerUnit }. Plot the values over the full grid to investigate.")
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
        fatal(s"Maximum likelihood estimate ${ math.exp(maxlogDelta) } for delta is not a global max. " +
          s"Plot the values over the full grid to investigate.")

      // fitting parabola logLkhd ~ a * x^2 + b * x + c near MLE by Lagrange interpolation gives
      // a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x2 - x1) * (x1 - x3) * (x3 - x2))
      // comparing to normal approx: logLkhd ~ 1 / (-2 * sigma^2) * x^2 + lower order terms:
      val sigmaH2 =
      math.sqrt(((x2 - x1) * (x1 - x3) * (x3 - x2)) / (-2 * (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2))))

      val h2LogLkhd = (0.01 to 0.99 by 0.01).map(h2 => logLkhdFunction.value(math.log((1 - h2) / h2)))

      val h2Lkhd = h2LogLkhd.map(ll => math.exp(ll - maxLogLkhd))
      val h2LkhdSum = h2Lkhd.sum
      val h2NormLkhd = FastIndexedSeq(Double.NaN) ++ h2Lkhd.map(_ / h2LkhdSum) ++ FastIndexedSeq(Double.NaN)

      (FastMath.exp(maxlogDelta), GlobalFitLMM(maxLogLkhd, gridLogLkhd, sigmaH2, h2NormLkhd))
    }

    def fitUsingDelta(delta: Double, optGlobalFit: Option[GlobalFitLMM]): DiagLMM = {
      val invDelta = 1 / delta
      val invD = (S + delta).map(1 / _)

      val Z = invD - invDelta

      val ydy = invDelta * yty + (Uty dot (Uty *:* Z))
      val Cdy = invDelta * Cty + (UtC.t * (Uty *:* Z))
      val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) *:* Z))

      val b = CdC \ Cdy
      val s2 = (ydy - (Cdy dot b)) / (if (useML) n else n - d)
      val sqrtInvD = sqrt(invD)
      val TC = UtC(::, *) *:* sqrtInvD
      val Ty = Uty *:* sqrtInvD
      val TyTy = Ty dot Ty

      DiagLMM(b, s2, math.log(s2), delta, optGlobalFit, sqrtInvD, TC, Ty, TyTy, useML)
    }

    val (delta, optGlobalFit) = optDelta match {
      case Some(delta0) => (delta0, None)
      case None =>
        val (delta0, gf) = fitDelta()
        (delta0, Some(gf))
    }

    fitUsingDelta(delta, optGlobalFit)
  }
}

case class GlobalFitLMM(maxLogLkhd: Double, gridLogLkhd: IndexedSeq[(Double, Double)], sigmaH2: Double, h2NormLkhd: IndexedSeq[Double])

case class DiagLMM(
  globalB: BDV[Double],
  globalS2: Double,
  logNullS2: Double,
  delta: Double,
  optGlobalFit: Option[GlobalFitLMM],
  sqrtInvD: BDV[Double],
  TC: BDM[Double],
  Ty: BDV[Double],
  TyTy: Double,
  useML: Boolean)

object LMMConstants {
  def apply(y: BDV[Double], C: BDM[Double],
    S: BDV[Double], Ut: BDM[Double]): LMMConstants = {
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

case class LMMConstants(y: BDV[Double], C: BDM[Double], S: BDV[Double], Ut: BDM[Double],
  Uty: BDV[Double], UtC: BDM[Double], Cty: BDV[Double],
  CtC: BDM[Double], yty: Double, n: Int, d: Int)
