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
    ("pval", TDouble),
    ("AF", TDouble),
    ("nHomRef", TInt),
    ("nHet", TInt),
    ("nHomVar", TInt),
    ("nMissing", TInt))

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
    val sampleMask = assocVds.sampleIds.map(completeSamplesSet).toArray

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
          Annotation(gf.sigmaH2, gf.maxLogLkhd, logDeltaGrid, logLkhdVals),
          TStruct(("seH2", TDouble), ("maxLogLkhd", TDouble), ("logDeltaGrid", TArray(TDouble)), ("logLkhdVals", TArray(TDouble))), rootGA + ".fit")
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
        val SparseGtVectorAndStats(x0, isConstant, af, nHomRef, nHet, nHomVar, nMissing) =
          RegressionUtils.toLinMixedHardCallStats(gs, sampleMaskBc.value, n)

        val lmmregAnnot =
          if (!isConstant) {
            val x: Vector[Double] = if (af <= sparsityThreshold) x0 else x0.toDenseVector
            val (b, s2, chi2, p) = scalerLMMBc.value.likelihoodRatioTest(TBc.value * x)

            Annotation(b, s2, chi2, p, af, nHomRef, nHet, nHomVar, nMissing)
          } else
            null

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
        val D = S + FastMath.exp(logDelta)
        val dy = y :/ D
        val ydy = y dot dy
        val Cdy = C.t * dy
        val CdC = C.t * (C(::, *) :/ D)
        val b = CdC \ Cdy
        val r = ydy - (Cdy dot b)

        -0.5 * (sum(breeze.numerics.log(D)) + n * math.log(r) + shift)
      }
    }

    object LogLkhdREML extends UnivariateFunction {
      val shift: Double = n * (1 + math.log(1d / n)) + (n - c) * math.log(2 * math.Pi) - logdet(C.t * C)._2

      def value(logDelta: Double): Double = {
        val D = S + FastMath.exp(logDelta)
        val dy = y :/ D
        val ydy = y dot dy
        val Cdy = C.t * dy
        val CdC = C.t * (C(::, *) :/ D)
        val b = CdC \ Cdy
        val r = ydy - (Cdy dot b)

        -0.5 * (sum(breeze.numerics.log(D)) + (n - c) * math.log(r) + logdet(CdC)._2 + shift)
      }
    }

    val logMin = -10
    val logMax = 10
    val pointsPerUnit = 100 // number of points per unit of log space

    val grid = (logMin * pointsPerUnit to logMax * pointsPerUnit).map(_.toDouble / pointsPerUnit) // avoids rounding of (logMin to logMax by logres)

    val gridLogLkhd = if (useML)
      grid.map(logDelta => (logDelta, LogLkhdML.value(logDelta)))
    else
      grid.map(logDelta => (logDelta, LogLkhdREML.value(logDelta)))

    val header = "logDelta\tlogLkhd"
    val gridValsString = gridLogLkhd.map{ case (d, nll) => s"$d\t$nll" }.mkString("\n")
    log.info(s"\nlmmreg: table of delta\n$header\n$gridValsString\n")

    val approxLogDelta = gridLogLkhd.maxBy(_._2)._1

    if (approxLogDelta == logMin)
      fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta lower search boundary e^$logMin = ${FastMath.exp(logMin)}, indicating negligible enviromental component of variance. The model is likely ill-specified.")
    else if (approxLogDelta == logMax)
      fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta upper search boundary e^$logMax = ${FastMath.exp(logMax)}, indicating negligible genetic component of variance. Standard linear regression may be more appropriate.")

    val searchInterval = new SearchInterval(-10d, 10d, approxLogDelta)
    val goal = GoalType.MAXIMIZE
    val objectiveFunction = new UnivariateObjectiveFunction(if (useML) LogLkhdML else LogLkhdREML)
    val brentOptimizer = new BrentOptimizer(5e-8, 5e-7) // tol = 5e-8 * abs((ln(delta))) + 5e-7 <= 1e-6
    val logDeltaPointValuePair = brentOptimizer.optimize(objectiveFunction, goal, searchInterval, MaxEval.unlimited)

    val logDelta = logDeltaPointValuePair.getPoint
    val maxLogLkhd = logDeltaPointValuePair.getValue

    if (math.abs(logDelta - approxLogDelta) > 1.0 / pointsPerUnit) {
      warn(s"lmmreg: the difference between the optimal value $approxLogDelta of ln(delta) on the grid and the optimal value $logDelta of ln(delta) by Brent's method exceeds the grid resolution of ${1.0 / pointsPerUnit}. Plot the values over the full grid to investigate.")
    }

    val indexBelow = (pointsPerUnit * (logDelta - logMin)).toInt
    val indexAbove = indexBelow + 1

    // three values of h2 = sigmoid(-ln(delta)) below, at, and above the MLE
    val x1 = sigmoid(-gridLogLkhd(indexBelow)._1)
    val x2 = sigmoid(-logDelta)
    val x3 = sigmoid(-gridLogLkhd(indexAbove)._1)

    assert(x1 > x2 && x2 > x3)

    // corresponding values of logLkhd
    val y1 = gridLogLkhd(indexBelow)._2
    val y2 = maxLogLkhd
    val y3 = gridLogLkhd(indexAbove)._2

    // fitting parabola logLkhd ~ a * x^2 + b * x + c near MLE by Lagrange interpolation
    val a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x2 - x1) * (x1 - x3) * (x3 - x2))

    // comparing to normal approx: logLkhd ~ 1 / (-2 * sigma^2) * x^2 + lower order terms
    val sigmaH2 = math.sqrt(1 / (-2 * a))

    (FastMath.exp(logDelta), GlobalFitLMM(maxLogLkhd, gridLogLkhd, sigmaH2))
  }
}

case class GlobalFitLMM(maxLogLkhd: Double, gridLogLkhd: IndexedSeq[(Double, Double)], sigmaH2: Double)

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

  def likelihoodRatioTest(x: Vector[Double]): (Double, Double, Double, Double) = {

    val n = y.length
    val Qtx = Qt * x
    val xQtx: Double = (x dot x) - (Qtx dot Qtx)
    val xQty: Double = (x dot y) - (Qtx dot Qty)

    val b: Double = xQty / xQtx
    val s2 = (yQty - xQty * b) / (if (useML) n else n - Qt.rows)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    (b, s2, chi2, p)
  }
}
