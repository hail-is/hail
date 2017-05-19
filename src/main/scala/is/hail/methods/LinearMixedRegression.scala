package is.hail.methods

import breeze.linalg._
import breeze.numerics.{sigmoid, sqrt}
import is.hail.annotations._
import is.hail.expr._
import is.hail.stats._
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
    sparsityThreshold: Double): VariantDataset = {

    require(assocVds.wasSplit)

    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(assocVds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = assocVds.stringSampleIds.map(s => completeSamplesSet(s)).toArray

    optDelta.foreach(delta =>
      if (delta <= 0d)
        fatal(s"delta must be positive, got ${ delta }"))

    val covNames = "intercept" +: covExpr

    val filteredKinshipMatrix = kinshipMatrix.filterSamples(completeSamplesSet)

    if (!(filteredKinshipMatrix.sampleIds sameElements completeSamples))
      fatal("Array of sample IDs in assoc_vds and array of sample IDs in kinship_matrix (with both filtered to complete " +
        "samples in assoc_vds) do not agree. This should not happen when kinship_vds is formed by filtering variants on assoc_vds.")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${plural(k, "covariate")} including intercept implies $d degrees of freedom.")

    info(s"lmmreg: running lmmreg on $n samples with $k sample ${plural(k, "covariate")} including intercept...")

    val cols = filteredKinshipMatrix.matrix.numCols().toInt

    val rrm = new DenseMatrix[Double](cols, cols, filteredKinshipMatrix.matrix.toBlockMatrix().toLocalMatrix().toArray)

    info(s"lmmreg: Computing eigenvectors of RRM...")

    val eigK = eigSymD(rrm)
    val Ut = eigK.eigenvectors.t
    val S = eigK.eigenvalues // increasing order

    assert(S.length == n)

    info("lmmreg: 20 largest evals: " + ((n - 1) to math.max(0, n - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info("lmmreg: 20 smallest evals: " + (0 until math.min(n, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    optDelta match {
      case Some(_) => info(s"lmmreg: Delta specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    val UtC = Ut * cov
    val Uty = Ut * y

    val diagLMM = DiagLMM(UtC, Uty, S, optDelta, useML)

    val delta = diagLMM.delta
    val globalBetaMap = covNames.zip(diagLMM.globalB.toArray).toMap
    val globalSg2 = diagLMM.globalS2
    val globalSe2 = delta * globalSg2
    val h2 = 1 / (1 + delta)

    val header = "rank\teval"
    val evalString = (0 until n).map(i => s"$i\t${ S(n - i - 1) }").mkString("\n")
    log.info(s"\nlmmreg: table of eigenvalues\n$header\n$evalString\n")

    info(s"lmmreg: global model fit: beta = $globalBetaMap")
    info(s"lmmreg: global model fit: sigmaG2 = $globalSg2")
    info(s"lmmreg: global model fit: sigmaE2 = $globalSe2")
    info(s"lmmreg: global model fit: delta = $delta")
    info(s"lmmreg: global model fit: h2 = $h2")

    diagLMM.optGlobalFit.foreach { gf =>
      info(s"lmmreg: global model fit: seH2 = ${gf.sigmaH2}")
    }

    val vds1 = assocVds.annotateGlobal(
      Annotation(useML, globalBetaMap, globalSg2, globalSe2, delta, h2, S.data.reverse: IndexedSeq[Double]),
      TStruct(("useML", TBoolean), ("beta", TDict(TString, TDouble)), ("sigmaG2", TDouble), ("sigmaE2", TDouble),
        ("delta", TDouble), ("h2", TDouble), ("evals", TArray(TDouble))), rootGA)

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
      info(s"lmmreg: Computing statistics for each variant...")

      val T = Ut(::, *) :* diagLMM.sqrtInvD
      val Qt = qr.reduced.justQ(diagLMM.TC).t
      val QtTy = Qt * diagLMM.Ty
      val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)

      val sc = assocVds.sparkContext
      val TBc = sc.broadcast(T)
      val sampleMaskBc = sc.broadcast(sampleMask)
      val scalerLMMBc = sc.broadcast(ScalerLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML))

      val (newVAS, inserter) = vds2.insertVA(LinearMixedRegression.schema, pathVA)

      vds2.mapAnnotations { case (v, va, gs) =>
        val optStats = RegressionUtils.toSparseHardCallStats(gs, sampleMaskBc.value, n)

        val lmmregAnnot = optStats.map { case (x0, af) =>
          val x: Vector[Double] = if (af <= sparsityThreshold) x0 else x0.toDenseVector

          scalerLMMBc.value.likelihoodRatioTest(TBc.value * x)
        }.orNull

        val newAnnotation = inserter(va, lmmregAnnot)
        assert(newVAS.typeCheck(newAnnotation))
        newAnnotation
      }.copy(vaSignature = newVAS)
    }
    else
      vds2
  }
}

object DiagLMM {
  def apply(
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    S: DenseVector[Double],
    optDelta: Option[Double] = None,
    useML: Boolean = false): DiagLMM = {

    require(C.rows == y.length)

    val (delta, optGlobalFit) = optDelta match {
        case Some(d) => (d, None)
        case None =>
          val (d, gf) = fitDelta(C, y, S, useML)
          (d, Some(gf))
      }

    val n = y.length
    val sqrtInvD = sqrt(S + delta).map(1 / _)
    val TC = C(::, *) :* sqrtInvD
    val Ty = y :* sqrtInvD
    val TyTy = Ty dot Ty
    val TCTy = TC.t * Ty
    val TCTC = TC.t * TC
    val b = TCTC \ TCTy
    val s2 = (TyTy - (TCTy dot b)) / (if (useML) n else n - C.cols)

    DiagLMM(b, s2, math.log(s2), delta, optGlobalFit, sqrtInvD, TC, Ty, TyTy, useML)
  }

  def fitDelta(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], useML: Boolean): (Double, GlobalFitLMM) = {

    val n = y.length
    val c = C.cols

    object LogLkhdML extends UnivariateFunction {
      val shift: Double = n * (1 + math.log(2 * math.Pi) + math.log(1d / n))

      def value(logDelta: Double): Double = {
        val invD = (S + FastMath.exp(logDelta)).map(1 / _)
        val dy = y :* invD
        val ydy = y dot dy
        val Cdy = C.t * dy
        val CdC = C.t * (C(::, *) :* invD)
        val b = CdC \ Cdy
        val r = ydy - (Cdy dot b)

        -0.5 * (-sum(breeze.numerics.log(invD)) + n * math.log(r) + shift)
      }
    }

    object LogLkhdREML extends UnivariateFunction {
      val shift: Double = n * (1 + math.log(1d / n)) + (n - c) * math.log(2 * math.Pi) - logdet(C.t * C)._2

      def value(logDelta: Double): Double = {
        val invD = (S + FastMath.exp(logDelta)).map(1 / _)
        val dy = y :* invD
        val ydy = y dot dy
        val Cdy = C.t * dy
        val CdC = C.t * (C(::, *) :* invD)
        val b = CdC \ Cdy
        val r = ydy - (Cdy dot b)

        -0.5 * (-sum(breeze.numerics.log(invD)) + (n - c) * math.log(r) + logdet(CdC)._2 + shift)
      }
    }

    val minLogDelta = -8
    val maxLogDelta = 8
    val pointsPerUnit = 100 // number of points per unit of log space

    val grid = (minLogDelta * pointsPerUnit to maxLogDelta * pointsPerUnit).map(_.toDouble / pointsPerUnit) // avoids rounding of (minLogDelta to logMax by logres)

    val gridLogLkhd = if (useML)
      grid.map(logDelta => (logDelta, LogLkhdML.value(logDelta)))
    else
      grid.map(logDelta => (logDelta, LogLkhdREML.value(logDelta)))

    val header = "logDelta\tlogLkhd"
    val gridValsString = gridLogLkhd.map{ case (d, nll) => s"$d\t$nll" }.mkString("\n")
    log.info(s"\nlmmreg: table of delta\n$header\n$gridValsString\n")

    val approxLogDelta = gridLogLkhd.maxBy(_._2)._1

    if (approxLogDelta == minLogDelta)
      fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta lower search boundary e^$minLogDelta = ${FastMath.exp(minLogDelta)}, indicating negligible enviromental component of variance. The model is likely ill-specified.")
    else if (approxLogDelta == maxLogDelta)
      fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta upper search boundary e^$maxLogDelta = ${FastMath.exp(maxLogDelta)}, indicating negligible genetic component of variance. Standard linear regression may be more appropriate.")

    val searchInterval = new SearchInterval(minLogDelta, maxLogDelta, approxLogDelta)
    val goal = GoalType.MAXIMIZE
    val objectiveFunction = new UnivariateObjectiveFunction(if (useML) LogLkhdML else LogLkhdREML)
    val brentOptimizer = new BrentOptimizer(5e-8, 5e-7) // tol = 5e-8 * abs((ln(delta))) + 5e-7 <= 1e-6
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
    val y1 = if (useML) LogLkhdML.value(z1) else LogLkhdREML.value(z1)
    val y2 = maxLogLkhd
    val y3 = if (useML) LogLkhdML.value(z3) else LogLkhdREML.value(z3)

    if (y1 >= y2 || y3 >= y2)
      fatal(s"Maximum likelihood estimate ${ math.exp(maxlogDelta) } for delta is not a global max." +
        s"Plot the values over the full grid to investigate.")

    // Fitting parabola logLkhd ~ a * x^2 + b * x + c near MLE by Lagrange interpolation gives
    // a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x2 - x1) * (x1 - x3) * (x3 - x2))
    // Comparing to normal approx: logLkhd ~ 1 / (-2 * sigma^2) * x^2 + lower order terms:
    val sigmaH2 =
      math.sqrt(((x2 - x1) * (x1 - x3) * (x3 - x2)) / (-2 * (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2))))

    val h2LogLkhd =
      if (useML)
        (0.01 to 0.99 by 0.01).map(h2 => LogLkhdML.value(math.log((1 - h2) / h2)))
      else
        (0.01 to 0.99 by 0.01).map(h2 => LogLkhdREML.value(math.log((1 - h2) / h2)))

    val h2Lkhd = h2LogLkhd.map(ll => math.exp(ll - maxLogLkhd))
    val h2LkhdSum = h2Lkhd.sum
    val h2NormLkhd = IndexedSeq(Double.NaN) ++ h2Lkhd.map(_ / h2LkhdSum) ++ IndexedSeq(Double.NaN)

    (FastMath.exp(maxlogDelta), GlobalFitLMM(maxLogLkhd, gridLogLkhd, sigmaH2, h2NormLkhd))
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

case class ScalerLMM(
  y: DenseVector[Double],
  yy: Double,
  Qt: DenseMatrix[Double],
  Qty: DenseVector[Double],
  yQty: Double,
  logNullS2: Double,
  useML: Boolean) {

  def likelihoodRatioTest(x: Vector[Double]): Annotation = {

    val n = y.length
    val Qtx = Qt * x
    val xQtx: Double = (x dot x) - (Qtx dot Qtx)
    val xQty: Double = (x dot y) - (Qtx dot Qty)

    val b: Double = xQty / xQtx
    val s2 = (yQty - xQty * b) / (if (useML) n else n - Qt.rows)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    Annotation(b, s2, chi2, p)
  }
}
