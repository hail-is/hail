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
    optNEigs: Option[Int] = None,
    droppedVarianceFraction: Double): VariantDataset = {

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

    val K = new DenseMatrix[Double](cols, cols, filteredKinshipMatrix.matrix.toBlockMatrixDense().toLocalMatrix().toArray)

    info(s"lmmreg: Computing eigenvectors of RRM...")

    val eigK = eigSymD(K)
    val fullU = eigK.eigenvectors
    val fullS = eigK.eigenvalues // increasing order
    val fullNEigs = fullS.length

    assert(fullS.length == n)

    optDelta match {
      case Some(_) => info(s"lmmreg: Delta specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    var nEigs = optNEigs.getOrElse(n)

    require(nEigs > 0 && nEigs <= fullS.length, s"lmmreg: Must specify number of eigenvectors between 1 and ${fullS.length}")

    nEigs = computeNEigs(fullS, nEigs, droppedVarianceFraction)

    val U = fullU(::, (fullNEigs - nEigs) until fullNEigs)

    val S = fullS((fullNEigs - nEigs) until fullNEigs)


    info(s"lmmreg: Evals 1 to ${math.min(20, nEigs)}: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Evals $nEigs to ${math.max(1, nEigs - 20)}: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val Ut = U.t

    val diagLMM = DiagLMM(y, cov, S, U, optDelta, useML)

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
      info(s"lmmreg: global model fit: seH2 = ${gf.sigmaH2}")
    }

    val vds1 = assocVds.annotateGlobal(
      Annotation(useML, globalBetaMap, globalSg2, globalSe2, delta, h2, fullS.data.reverse: IndexedSeq[Double], nEigs),
      TStruct(("useML", TBoolean), ("beta", TDict(TString, TDouble)), ("sigmaG2", TDouble), ("sigmaE2", TDouble),
        ("delta", TDouble), ("h2", TDouble), ("evals", TArray(TDouble)), ("nEigs", TInt)), rootGA)

    val vds2 = diagLMM.optGlobalFit match {
      case Some(gf) =>
        val (logDeltaGrid, logLkhdVals) = gf.gridLogLkhd.unzip
        vds1.annotateGlobal(
          Annotation(gf.sigmaH2, gf.h2NormLkhd, gf.maxLogLkhd, logDeltaGrid, logLkhdVals, nEigs),
          TStruct(("seH2", TDouble), ("normLkhdH2", TArray(TDouble)), ("maxLogLkhd", TDouble),
            ("logDeltaGrid", TArray(TDouble)), ("logLkhdVals", TArray(TDouble))), rootGA + ".fit")
      case None =>
        assert(optDelta.isDefined)
        vds1
    }



    if (runAssoc) {
      
      val sc = assocVds.sparkContext
      val sampleMaskBc = sc.broadcast(sampleMask)
      val (newVAS, inserter) = vds2.insertVA(LinearMixedRegression.schema, pathVA)

      info(s"lmmreg: Computing statistics for each variant...")

      if (false) {
        val T = Ut(::, *) :* diagLMM.sqrtInvD
        val Qt = qr.reduced.justQ(diagLMM.TC).t
        val QtTy = Qt * diagLMM.Ty
        val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)

        val TBc = sc.broadcast(T)
        val scalerLMMBc = sc.broadcast(ScalerLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML))


        vds2.mapAnnotations { case (v, va, gs) =>
          val x: Vector[Double] =
            if (!useDosages) {
              val x0 = RegressionUtils.hardCalls(gs, n, sampleMaskBc.value)
              if (x0.used <= sparsityThreshold * n) x0 else x0.toDenseVector
            }
            else
              RegressionUtils.dosages(gs, n, sampleMaskBc.value)

          // TODO constant checking to be removed in 0.2
          val nonConstant = useDosages || !RegressionUtils.constantVector(x)

          val lmmregAnnot = if (nonConstant) scalerLMMBc.value.likelihoodRatioTest(TBc.value * x) else null
          val newAnnotation = inserter(va, lmmregAnnot)
          assert(newVAS.typeCheck(newAnnotation))
          newAnnotation
        }.copy(vaSignature = newVAS)
      }
      else {

        val slowLMMBc = sc.broadcast(SlowLMM(y, cov, S, U, delta, diagLMM.logNullS2, useML))

        vds2.mapAnnotations { case (v, va, gs) =>
          val x: Vector[Double] =
            if (!useDosages) {
              val x0 = RegressionUtils.hardCalls(gs, n, sampleMaskBc.value)
              if (x0.used <= sparsityThreshold * n) x0 else x0.toDenseVector
            }
            else
              RegressionUtils.dosages(gs, n, sampleMaskBc.value)

          val nonConstant = useDosages || !RegressionUtils.constantVector(x)

          val lmmregAnnot = if (nonConstant) slowLMMBc.value.liklihoodRatioTest(x) else null
          val newAnnotation = inserter(va, lmmregAnnot)
          assert(newVAS.typeCheck(newAnnotation))
          newAnnotation
        }.copy(vaSignature = newVAS)
      }
    }
    else
      vds2
  }

  def computeNEigs(S: DenseVector[Double], nEigs: Int, droppedVarianceFraction: Double): Int = {
    val trace = S.toArray.sum
    var i = S.length - 1
    var runningSum = 0.0
    val target = droppedVarianceFraction * trace
    while (i > nEigs || runningSum <= target) {
      runningSum += S(-i)
      i -= 1
    }
    math.min(nEigs, i + 1)
  }
}

case class SlowLMM(y: DenseVector[Double], covs: DenseMatrix[Double], S: DenseVector[Double], U: DenseMatrix[Double], delta: Double, logNullS2: Double, useML: Boolean) {

  val n = y.length
  val d = covs.cols
  val D = S + delta
  val Ut = U.t
  val Uty = Ut * y
  val UtCov = Ut * covs
  val yty = y.t * y
  val dy = Uty :/ D

  val covTcov = covs.t * covs
  val covTy = covs.t * y

  val invDMinusInvDelta = D.map(x => 1 / x) - 1 / delta
  val ydy = yty / delta +
    (Uty dot (Uty :* invDMinusInvDelta))

  val k = S.length


  def liklihoodRatioTest(v: Vector[Double]): Annotation = {
    val vty = v dot y
    val vtv = v dot v
    val covtv = covs.t * v
    val Utv = Ut * v

    val UtC = DenseMatrix.horzcat(Utv.toDenseVector.toDenseMatrix.t, UtCov)

    val CtC = DenseMatrix.vertcat(DenseVector.vertcat(DenseVector(vtv), covtv).toDenseVector.toDenseMatrix,
      DenseMatrix.horzcat(covtv.toDenseVector.toDenseMatrix.t, covTcov))
    val Cty = DenseVector.vertcat(DenseVector(vty), covTy)

    val Cdy = Cty / delta +
      (UtC.t  * (Uty :* invDMinusInvDelta))

    val ZUtv = Utv :* invDMinusInvDelta
    val ZUtCov = UtCov(::, *) :* invDMinusInvDelta//Rowwise.

    //4 Parts
    val topLeft = Utv dot ZUtv
    val bottomLeft = UtCov.t * ZUtv
    val topRight = Utv.t * ZUtCov
    val bottomRight = UtCov.t * ZUtCov

    val top = DenseVector.vertcat(DenseVector(topLeft), topRight.t)
    val bottom = DenseMatrix.horzcat(bottomLeft.toDenseMatrix.t, bottomRight)

    val CdC = CtC / delta + DenseMatrix.vertcat(top.toDenseVector.toDenseMatrix, bottom)

    val b = CdC \ Cdy

    val r2 = ydy - (Cdy dot b)

    val s2 = r2 / (if (useML) n else n - d)

    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    Annotation(b(0), s2, chi2, p)
  }

}

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


object DiagLMM {
  def apply(
    y: DenseVector[Double],
    C: DenseMatrix[Double],
    S: DenseVector[Double],
    U: DenseMatrix[Double],
    optDelta: Option[Double] = None,
    useML: Boolean = false): DiagLMM = {

    new DiagLMMSolver(C, y, S, U, optDelta, useML).solve()
  }
}


class DiagLMMSolver(
  C: DenseMatrix[Double],
  y: DenseVector[Double],
  S: DenseVector[Double],
  U: DenseMatrix[Double],
  optDelta: Option[Double] = None,
  useML: Boolean = false) {

  val Ut = U.t
  val UtC = Ut * C
  val Uty = Ut * y

  val CtC = C.t * C
  val Cty = C.t * y
  val yty = y.t * y

  val n = y.length
  val d = C.cols

  //MR indicates difference between thing and it's rotated version.
  val ytyMR = (yty - Uty.t * Uty)
  val CtyMR = (Cty - UtC.t * Uty)
  val CtCMR = (CtC - UtC.t * UtC)

  val (delta, optGlobalFit) = optDelta match {
    case Some(del) => (del, None)
    case None =>
      val (del, gf) = fitDelta()
      (del, Some(gf))
  }

  def solve(): DiagLMM = {
    val invD = (S + delta).map(1 / _)
    val dy = Uty :* invD
    var ydy = Uty dot dy
    var Cdy = UtC.t * dy
    var CdC = UtC.t * (UtC(::, *) :* invD)

    if(S.length < n) {
      ydy = ydy + ytyMR / delta
      Cdy = Cdy + CtyMR / delta
      CdC = CdC + CtCMR / delta
    }

    val b = CdC \ Cdy

    val r2 = ydy - (Cdy dot b)

    val denom = if (useML) n else n - d

    val s2 = r2 / denom
    val sqrtInvD = sqrt(S + delta).map(1 / _)
    val TC = UtC(::, *) :* sqrtInvD
    val Ty = Uty :* sqrtInvD
    val TyTy = Ty dot Ty

    DiagLMM(b, s2, math.log(s2), delta, optGlobalFit, sqrtInvD, TC, Ty, TyTy, useML)
  }

  def fitDelta(): (Double, GlobalFitLMM) = {

    object LogLkhdML extends UnivariateFunction {
      val shift = -0.5 * n * (1 + math.log(2 * math.Pi))

      def value(logDelta: Double): Double = {
        val delta = FastMath.exp(logDelta)
        val D = S + delta
        val dy = Uty :/ D

        val invDMinusInvDelta = D.map(x => 1 / x) - 1 / delta

        val ydy = yty / delta +
          (Uty dot (Uty :* invDMinusInvDelta))

        val Cdy = Cty / delta +
          (UtC.t  * (Uty :* invDMinusInvDelta))

        val CdC = CtC / delta +
          (UtC.t * (UtC(::, *) :* invDMinusInvDelta))

        val k = S.length
        val b = CdC \ Cdy

        val logdetD = sum(breeze.numerics.log(D)) + (n - k) * logDelta

        val r2 = ydy - (Cdy dot b)

        val sigma2 = r2 / n

        -0.5 * (logdetD + n * (math.log(sigma2))) + shift
      }

    }

    object LogLkhdREML extends UnivariateFunction {
      val shift = -0.5 * (n - d) * (1 + math.log(2 * math.Pi))

      def value(logDelta: Double): Double = {
        val delta = FastMath.exp(logDelta)
        val D = S + delta
        val dy = Uty :/ D
        val ydy =
          (Uty dot dy) +
          ytyMR / delta
        val Cdy =
          UtC.t * dy +
          CtyMR / delta
        val CdC =
          UtC.t * (UtC(::, *) :/ D) +
          CtCMR / delta
        
        val b = CdC \ Cdy
        
        val k = S.length

        val r2 = ydy - (Cdy dot b)
        val sigma2 = r2 / (n - d)

        val logdetD = sum(breeze.numerics.log(D)) + (n - k) * logDelta
        val logdetCdC = logdet(CdC)._2
        val logdetCtC = logdet(CtC)._2

        -0.5 * (logdetD + logdetCdC - logdetCtC + (n - d) * (math.log(sigma2))) + shift
      }
    }

    val minLogDelta = -8
    val maxLogDelta = 8
    val pointsPerUnit = 100 // number of points per unit of log space

    val grid = (minLogDelta * pointsPerUnit to maxLogDelta * pointsPerUnit).map(_.toDouble / pointsPerUnit) // avoids rounding of (minLogDelta to logMax by logres)
    val logLkhdFunction = if (useML) LogLkhdML else LogLkhdREML

    val gridLogLkhd = grid.map(logDelta => (logDelta, logLkhdFunction.value(logDelta)))

    val header = "logDelta\tlogLkhd"
    val gridValsString = gridLogLkhd.map { case (d, nll) => s"$d\t$nll" }.mkString("\n")
    log.info(s"\nlmmreg: table of delta\n$header\n$gridValsString\n")

    val approxLogDelta = gridLogLkhd.maxBy(_._2)._1

    if (approxLogDelta == minLogDelta)
      fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta lower search boundary e^$minLogDelta = ${FastMath.exp(minLogDelta)}, indicating negligible enviromental component of variance. The model is likely ill-specified.")
    else if (approxLogDelta == maxLogDelta)
      fatal(s"lmmreg: failed to fit delta: ${if (useML) "ML" else "REML"} realized at delta upper search boundary e^$maxLogDelta = ${FastMath.exp(maxLogDelta)}, indicating negligible genetic component of variance. Standard linear regression may be more appropriate.")

    val searchInterval = new SearchInterval(minLogDelta, maxLogDelta, approxLogDelta)
    val goal = GoalType.MAXIMIZE
    val objectiveFunction = new UnivariateObjectiveFunction(logLkhdFunction)
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
    val y1 = logLkhdFunction.value(z1)
    val y2 = maxLogLkhd
    val y3 = logLkhdFunction.value(z3)

    if (y1 >= y2 || y3 >= y2)
      fatal(s"Maximum likelihood estimate ${ math.exp(maxlogDelta) } for delta is not a global max. " +
        s"Plot the values over the full grid to investigate.")

    // Fitting parabola logLkhd ~ a * x^2 + b * x + c near MLE by Lagrange interpolation gives
    // a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x2 - x1) * (x1 - x3) * (x3 - x2))
    // Comparing to normal approx: logLkhd ~ 1 / (-2 * sigma^2) * x^2 + lower order terms:
    val sigmaH2 =
      math.sqrt(((x2 - x1) * (x1 - x3) * (x3 - x2)) / (-2 * (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2))))

    val h2LogLkhd = (0.01 to 0.99 by 0.01).map(h2 => logLkhdFunction.value(math.log((1 - h2) / h2)))


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


