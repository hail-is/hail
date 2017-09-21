package is.hail.methods

import is.hail.utils._
import is.hail.variant._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats.{LogisticRegressionModel, RegressionUtils, SkatModel}
import is.hail.annotations.Annotation
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object Skat {
  def apply(vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    yExpr: String,
    covExpr: Array[String],
    weightExpr: Option[String],
    logistic: Boolean,
    useDosages: Boolean,
    accuracy: Double,
    iterations: Int,
    forceLargeN: Boolean = false): KeyTable = { // useLargeN used to force runSkatPerKeyLargeN in testing
    val (y, cov, completeSampleIndex) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)

    if (accuracy <= 0)
      fatal(s"tolerance must be positive, default is 1e-6, got $accuracy")
    
    if (iterations <= 0)
      fatal(s"iterations must be positive, default is 10000, got $iterations")
    
    if (logistic) {
      val badVals = y.findAll(yi => yi != 0d && yi != 1d)
      if (badVals.nonEmpty)
        fatal(s"For logistic SKAT, phenotype must be Boolean or numeric with value 0 or 1 for each complete " +
          s"sample; found ${badVals.length} ${plural(badVals.length, "violation")} starting with ${badVals(0)}")
    }
    
    val n = y.size
    val sampleMask = Array.fill[Boolean](vds.nSamples)(false)
    completeSampleIndex.foreach(i => sampleMask(i) = true)

    val filteredVds = vds.filterSamplesMask(sampleMask)
    
    val (keyedRdd, keyType) = formKeyedRdd(filteredVds, variantKeys, singleKey, weightExpr, useDosages)

    val skatRdd: RDD[Row] = 
      if (logistic)
        logisticSkat(keyedRdd, keyType, y, cov, accuracy, iterations, forceLargeN)
      else
        linearSkat(keyedRdd, keyType, y, cov, accuracy, iterations, forceLargeN)
    
    val skatSignature = skatSchema(keyType)

    new KeyTable(vds.hc, skatRdd, skatSignature, Array("key"))
  }
  
  def skatSchema(keyType: Type) = TStruct(
    ("key", keyType),
    ("size", TInt32),
    ("qstat", TFloat64),
    ("pval", TFloat64),
    ("fault", TInt32))
  
  def formKeyedRdd(vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    useDosages: Boolean):
  (RDD[(Annotation, Iterable[(Vector[Double], Double)])], Type) = {
    
    val n = vds.nSamples
    
    val vdsWithWeight =
      if (weightExpr.isEmpty)
        vds.annotateVariantsExpr("va.__AF = gs.callStats(g => v).AF")
          .annotateVariantsExpr("va.__weight = let af = if (va.__AF[0] <= va.__AF[1]) va.__AF[0] else va.__AF[1] in dbeta(af, 1.0, 25.0)**2")
      else
        vds

    val (keysType, keysQuerier) = vdsWithWeight.queryVA(variantKeys)
    val (weightType, weightQuerier) = weightExpr match {
      case None => vdsWithWeight.queryVA("va.__weight")
      case Some(expr) => vdsWithWeight.queryVA(expr)
    }

    val typedWeightQuerier = weightType match {
      case _: TNumeric => (x: Annotation) => DoubleNumericConversion.to(weightQuerier(x))
      case _ => fatal("Weight must evaluate to numeric type")
    }

    val (keyType, keyIterator): (Type, Annotation => Iterator[Annotation]) = if (singleKey) {
      (keysType, (key: Annotation) => Iterator.single(key))
    } else {
      val keyType = keysType match {
        case t: TIterable => t.elementType
        case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
      }
      (keyType, (keys: Annotation) => keys.asInstanceOf[Iterable[Annotation]].iterator)
    }
    
    val completeSamplesBc = vds.sparkContext.broadcast((0 until n).toArray)

    (vdsWithWeight.rdd.flatMap { case (_, (va, gs)) =>
      (Option(keysQuerier(va)), Option(typedWeightQuerier(va))) match {
        case (Some(key), Some(w)) =>
          val gVector: Vector[Double] =
            if (!useDosages) {
              RegressionUtils.hardCalls(gs, n)
            } else {
              RegressionUtils.dosages(gs, completeSamplesBc.value)
            }
          keyIterator(key).map((_, (gVector, w)))
        case _ => Iterator.empty
      }
    }.groupByKey(), keyType)
  }
  
  def linearSkat(keyedRdd: RDD[(Annotation, Iterable[(Vector[Double], Double)])], keyType: Type,
    y: DenseVector[Double], cov: DenseMatrix[Double],
    accuracy: Double, iterations: Int, forceLargeN: Boolean): RDD[Row] = {
    
    val n = y.size
    val k = cov.cols
    val d = n - k

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    // fit null model
    val qr.QR(q, r) = qr.reduced.impl_reduced_DM_Double(cov)
    val beta = r \ (q.t * y)
    val res = y - cov * beta
    val sigmaSq = (res dot res) / d

    val sc = keyedRdd.sparkContext
    val resBc = sc.broadcast(res)
    val QtBc = sc.broadcast(q.t)

    def preprocessGenotypes(x: Vector[Double], w: Double): SkatTuple = {
      if (w < 0)
        fatal(s"Variant weights must be non-negative, got $w")
      val sqrtw = math.sqrt(w)
      val xw = x * sqrtw
      val sqrt_qi = resBc.value dot xw
      SkatTuple(sqrt_qi * sqrt_qi, xw, QtBc.value * xw)
    }

    keyedRdd
      .map { case (key, vs) =>
        val vArray = vs.map((preprocessGenotypes _).tupled).toArray
        val (q, gramian) = if (vArray.length.toLong * n < Int.MaxValue && !forceLargeN) {
          computeGramianSmallN(vArray)
        } else {
          computeGramianLargeN(vArray)
        }
        val (pval, fault) = SkatModel.computeStats(q / sigmaSq, gramian, accuracy, iterations)
        Row(key, vArray.length, q / (2 * sigmaSq), pval, fault)
      }
  }

  def logisticSkat(keyedRdd: RDD[(Any, Iterable[(Vector[Double], Double)])], keyType: Type,
    y: DenseVector[Double], cov: DenseMatrix[Double],
    accuracy: Double, iterations: Int, forceLargeN: Boolean): RDD[Row] = {

    val n = y.size
    val k = cov.cols
    val d = n - k

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    val logRegM = new LogisticRegressionModel(cov, y).fit()
    if (!logRegM.converged)
      fatal("Failed to fit logistic regression null model (MLE with covariates only): " + (
        if (logRegM.exploded)
          s"exploded at Newton iteration ${ logRegM.nIter }"
        else
          "Newton iteration failed to converge"))

    val mu = sigmoid(cov * logRegM.b)
    val V = mu.map(x => x * (1 - x))
    val VX = cov(::, *) :* V
    val XtVX = cov.t * VX
    XtVX.forceSymmetry()
    var Cinv: DenseMatrix[Double] = null
    try {
      Cinv = inv(cholesky(XtVX))
    } catch {
      case _: MatrixSingularException => fatal("Singular matrix exception while computing Cholesky factor of XtVX")
      case _: NotConvergedException => fatal("Inversion of Cholesky factor of XtVX did not converge")
    }
    val res = y - mu

    val sc = keyedRdd.sparkContext
    val sqrtVBc = sc.broadcast(sqrt(V))
    val resBc = sc.broadcast(res)
    val CinvXtVBc = sc.broadcast(Cinv * VX.t)

    def preprocessGenotypes(x: Vector[Double], w: Double): SkatTuple = {
      val xw = x * math.sqrt(w)
      val sqrt_qi = resBc.value dot xw
      val CinvXtVwx = CinvXtVBc.value * xw
      
      SkatTuple(sqrt_qi * sqrt_qi, xw :* sqrtVBc.value , CinvXtVwx)
    }

    keyedRdd.map { case (key, vs) =>
      val vArray = vs.map((preprocessGenotypes _).tupled).toArray
      val (q, gramian) = if (vArray.length.toLong * n < Int.MaxValue && !forceLargeN) {
        computeGramianSmallN(vArray)
      } else {
        computeGramianLargeN(vArray)
      }
      val (pval, fault) = SkatModel.computeStats(q, gramian, accuracy, iterations)
      
      Row(key, vArray.length, q / 2, pval, fault)
    }
  }

  /*
  Davies algorithm uses the eigenvalues of (G * sqrt(W)).t * P_0 * G * sqrt(W)
  We evaluate this matrix as A.t * A - B.t * B where
  linear:   A = G * sqrt(W)              B = Q.t * G * sqrt(W)
  logistic: A = sqrt(V) * G * sqrt(W)    B = inv(C) * X.t * V * G * sqrt(W)
  Here X = Q * R and C * C.t = X.t * V * X
  */
  def computeGramianSmallN(st: Array[SkatTuple]): (Double, DenseMatrix[Double]) = {
    require(st.nonEmpty)
    
    val m = st.length
    val n = st(0).xw.size    
    val k = st(0).Qtxw.size
    val isDenseVector = st(0).xw.isInstanceOf[DenseVector[Double]]
        
    val Adata = new Array[Double](m * n)
    var i = 0
    if (isDenseVector) {
      while (i < m) {
        val xwi = st(i).xw
        var j = 0
        while (j < n) {
          Adata(i * n + j) = xwi(j)
          j += 1
        }
        i += 1
      }
    } else {
      while (i < m) {
        val xwi = st(i).xw.asInstanceOf[SparseVector[Double]]
        val nnz = xwi.used
        var j = 0
        while (j < nnz) {
          val index = xwi.index(j)
          Adata(i * n + index) = xwi.data(j)
          j += 1
        }
        i += 1
      }
    }
    
    val Bdata = new Array[Double](k * m)
    var q = 0.0
    i = 0
    while (i < m) {
      q += st(i).qi
      val Qtxwi = st(i).Qtxw
      var j = 0
      while (j < k) {
        Bdata(i * k + j) = Qtxwi(j)
        j += 1
      }
      i += 1
    }

    val A = new DenseMatrix[Double](n, m, Adata)
    val B = new DenseMatrix[Double](k, m, Bdata)
    
    (q, A.t * A - B.t * B)
  }

  def computeGramianLargeN(st: Array[SkatTuple]): (Double, DenseMatrix[Double]) = {
    val m = st.length
    val data = Array.ofDim[Double](m * m)
    var q = 0.0

    var i = 0
    while (i < m) {
      q += st(i).qi
      val xwi = st(i).xw
      val Qtxwi = st(i).Qtxw
      data(i * (m + 1)) = (xwi dot xwi) - (Qtxwi dot Qtxwi)
      var j = 0
      while (j < i) {
        val temp = (xwi dot st(j).xw) - (Qtxwi dot st(j).Qtxw)
        data(i * m + j) = temp
        data(j * m + i) = temp
        j += 1
      }
      i += 1

    }
    
    (q, new DenseMatrix[Double](m, m, data))
  }
}

case class SkatTuple(qi: Double, xw: Vector[Double], Qtxw: DenseVector[Double])
