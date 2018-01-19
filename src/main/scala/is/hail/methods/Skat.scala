package is.hail.methods

import is.hail.utils._
import is.hail.variant._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.stats.{LogisticRegressionModel, RegressionUtils, eigSymD}
import is.hail.annotations.{Annotation, UnsafeRow}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference
import is.hail.expr.types.TStruct

/*
Skat implements the burden test described in:

Wu MC, Lee S, Cai T, Li Y, Boehnke M, Lin X.
Rare-Variant Association Testing for Sequencing Data with the Sequence Kernel Association Test.
American Journal of Human Genetics. 2011;89(1):82-93. doi:10.1016/j.ajhg.2011.05.029.

For n samples and a group of m variants, we have:
y = n x 1 vector of phenotypes
X = n x k matrix of covariates including intercept = cov
mu = n x 1 vector of predictions under the null model, linear: mu = Xb, logistic: mu = sigmoid(Xb)
W = m x m diagonal matrix of variant weights
G = n x m matrix of genotypes

The variance component score statistic in the paper is:
Q = (y - mu).t * G * W * G.t * (y - mu)

The null distribution of Q is a mixture of independent 1 d.o.f. chi-squared random variables
with coefficients given by the non-zero eigenvalues of n x n matrix
Z * Z.t = sqrt(P_0) * G * W * G.t * sqrt(P_0)
where
P_0 = V - V * X * (X.t * V * X)^-1 * X.t * V
V = n x n diagonal matrix with diagonal elements given by sigmaSq for linear and mu_i * (1 - mu_i) for logistic

To scale to large n, we exploit that Z * Z.t has the same non-zero eigenvalues as the m x m matrix
Z.t * Z = sqrt(W) * G.t * P_0 * G * sqrt(W)
and express the latter gramian matrix in terms of matrices A and B as follows:
linear:   sigmaSq * Z.t * Z = A.t * A - B.t * B,    A = G * sqrt(W)              B = Q0.t * G * sqrt(W)
logistic:           Z.t * Z = A.t * A - B.t * B,    A = sqrt(V) * G * sqrt(W)    B = C^-1 * X.t * V * G * sqrt(W)
where
Q0 = n x k matrix in QR decomposition of X = Q0 * R
C = k x k Cholesky factor of X.t * V * X = C * C.t

For each variant, SkatTuple encodes the corresponding summand of Q and columns of A and B.
We compute and group SkatTuples by key. Then, for each key, we compute Q and A.t * A - B.t * B,
the eigenvalues of the latter, and the p-value with the Davies algorithm.
*/
case class SkatTuple(q: Double, a: DenseVector[Double], b: DenseVector[Double])

object Skat {
  val hardMaxEntriesForSmallN = 64e6 // 8000 x 8000 => 512MB of doubles
  
  def apply(vsm: MatrixTable,
    keyExpr: String,
    weightExpr: String,
    yExpr: String,
    xExpr: String,
    covExpr: Array[String],
    logistic: Boolean,
    maxSize: Int,
    accuracy: Double,
    iterations: Int): Table = {
    
    if (maxSize <= 0 || maxSize > 46340)
      fatal(s"Maximum group size must be in [1, 46340], got $maxSize")

    val maxEntriesForSmallN = math.min(maxSize * maxSize, hardMaxEntriesForSmallN)

    if (accuracy <= 0)
      fatal(s"tolerance must be positive, default is 1e-6, got $accuracy")
    if (iterations <= 0)
      fatal(s"iterations must be positive, default is 10000, got $iterations")
    
    val (y, cov, completeSampleIndex) = RegressionUtils.getPhenoCovCompleteSamples(vsm, yExpr, covExpr)

    val n = y.size
    val k = cov.cols
    val d = n - k

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")    
    if (logistic) {
      val badVals = y.findAll(yi => yi != 0d && yi != 1d)
      if (badVals.nonEmpty)
        fatal(s"For logistic SKAT, phenotype must be Boolean or numeric with value 0 or 1 for each complete " +
          s"sample; found ${badVals.length} ${plural(badVals.length, "violation")} starting with ${badVals(0)}")
    }
    
    val (keyGsWeightRdd, keyType) =
      computeKeyGsWeightRdd(vsm, xExpr, completeSampleIndex, keyExpr, weightExpr)

    val sc = keyGsWeightRdd.sparkContext

    def linearSkat(): RDD[Row] = { 
      // fit null model
      val QR = qr.reduced(cov)
      val Qt = QR.q.t
      val R = QR.r
      val beta = R \ (Qt * y)
      val res = y - cov * beta
      val sigmaSq = (res dot res) / d
  
      val resBc = sc.broadcast(res)
      val QtBc = sc.broadcast(Qt)
      
      def linearTuple(x: DenseVector[Double], w: Double): SkatTuple = {
        val xw = x * math.sqrt(w)
        val sqrt_q = resBc.value dot xw
        SkatTuple(sqrt_q * sqrt_q, xw, QtBc.value * xw)
      }
            
      keyGsWeightRdd
        .map { case (key, vs) =>
          val vsArray = vs.toArray
          val size = vsArray.length
          if (size <= maxSize) {
            val skatTuples = vsArray.map((linearTuple _).tupled)
            val (q, gramian) = computeGramian(skatTuples, size.toLong * n <= maxEntriesForSmallN)
            
            // using q / sigmaSq since Z.t * Z = gramian / sigmaSq
            val (pval, fault) = computePval(q / sigmaSq, gramian, accuracy, iterations)
            
            // returning qstat = q / (2 * sigmaSq) to agree with skat R table convention
            Row(key, size, q / (2 * sigmaSq), pval, fault)
          } else {
            Row(key, size, null, null, null)
          }
        }
    }
        
    def logisticSkat(): RDD[Row] = {
      val logRegM = new LogisticRegressionModel(cov, y).fit()
      if (!logRegM.converged)
        fatal("Failed to fit logistic regression null model (MLE with covariates only): " + (
          if (logRegM.exploded)
            s"exploded at Newton iteration ${ logRegM.nIter }"
          else
            "Newton iteration failed to converge"))
  
      val mu = sigmoid(cov * logRegM.b)
      val V = mu.map(x => x * (1 - x))
      val VX = cov(::, *) *:* V
      val XtVX = cov.t * VX
      XtVX.forceSymmetry()
      var Cinv: DenseMatrix[Double] = null
      try {
        Cinv = inv(cholesky(XtVX))
      } catch {
        case e: MatrixSingularException =>
          fatal("Singular matrix exception while computing Cholesky factor of X.t * V * X.\n" + e.getMessage)
        case e: NotConvergedException =>
          fatal("Not converged exception while inverting Cholesky factor of X.t * V * X.\n" + e.getMessage)
      }
      val res = y - mu
  
      val sqrtVBc = sc.broadcast(sqrt(V))
      val resBc = sc.broadcast(res)
      val CinvXtVBc = sc.broadcast(Cinv * VX.t)
  
      def logisticTuple(x: DenseVector[Double], w: Double): SkatTuple = {
        val xw = x * math.sqrt(w)
        val sqrt_q = resBc.value dot xw      
        SkatTuple(sqrt_q * sqrt_q, xw *:* sqrtVBc.value , CinvXtVBc.value * xw)
      }
  
      keyGsWeightRdd.map { case (key, vs) =>
        val vsArray = vs.toArray
        val size = vsArray.length
        if (size <= maxSize) {
          val skatTuples = vs.map((logisticTuple _).tupled).toArray
          val (q, gramian) = computeGramian(skatTuples, size.toLong * n <= maxEntriesForSmallN)
          val (pval, fault) = computePval(q, gramian, accuracy, iterations)
  
          // returning qstat = q / 2 to agree with skat R table convention
          Row(key, size, q / 2, pval, fault)
        } else {
          Row(key, size, null, null, null)
        }
      }
    }
    
    val skatRdd = if (logistic) logisticSkat() else linearSkat()
    
    val skatSignature = TStruct(
      ("key", keyType),
      ("size", TInt32()),
      ("qstat", TFloat64()),
      ("pval", TFloat64()),
      ("fault", TInt32()))

    Table(vsm.hc, skatRdd, skatSignature, Array("key"))
  }

  def computeKeyGsWeightRdd(vsm: MatrixTable,
    xExpr: String,
    completeSampleIndex: Array[Int],
    keyExpr: String,
    // returns ((key, [(gs_v, weight_v)]), keyType)
    weightExpr: String): (RDD[(Annotation, Iterable[(DenseVector[Double], Double)])], Type) = {
    val ec = vsm.matrixType.genotypeEC
    val xf = RegressionUtils.parseExprAsDouble(xExpr, ec)

    val (keyType, keyQuerier) = vsm.queryVA(keyExpr)
    val (weightType, weightQuerier) = vsm.queryVA(weightExpr)

    val typedWeightQuerier = weightType match {
      case _: TNumeric => (a: Annotation) => Option(weightQuerier(a)).map(DoubleNumericConversion.to)
      case _ => fatal(s"Weight must have numeric type, got $weightType")
    }
    
    val sc = vsm.sparkContext

    val localGlobalAnnotationBc = sc.broadcast(vsm.globalAnnotation)
    val sampleIdsBc = vsm.sampleIdsBc
    val sampleAnnotationsBc = vsm.sampleAnnotationsBc
    val localRowType = vsm.rowType

    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)

    (vsm.rdd2.rdd.flatMap { rv =>
      val ur = new UnsafeRow(localRowType, rv)
      val va = ur.get(2)

      (Option(keyQuerier(va)), typedWeightQuerier(va)) match {
        case (Some(key), Some(w)) =>
          if (w < 0)
            fatal(s"Variant weights must be non-negative, got $w")

          val v = ur.get(1)
          val gs = ur.getAs[IndexedSeq[Annotation]](3)
          val n = completeSampleIndexBc.value.length
          val x = new DenseVector[Double](n)
          
          val missingSamples = new ArrayBuilder[Int]()

          RegressionUtils.inputVector(x,
            localGlobalAnnotationBc.value, sampleIdsBc.value, sampleAnnotationsBc.value, (v, (va, gs)),
            ec, xf,
            completeSampleIndexBc.value, missingSamples)

          Option((key, (x: DenseVector[Double], w)))
        case _ => None
      }
    }.groupByKey(), keyType)
  }
 
  
  def computeGramian(st: Array[SkatTuple], useSmallN: Boolean): (Double, DenseMatrix[Double]) =
    if (useSmallN) computeGramianSmallN(st) else computeGramianLargeN(st)

  def computeGramianSmallN(st: Array[SkatTuple]): (Double, DenseMatrix[Double]) = {
    require(st.nonEmpty)
    val st0 = st(0)
    
    // Holds for all st(i) by construction of linearTuple and logisticTuple, checking st(0) defensively
    require(st0.a.offset == 0 && st0.a.stride == 1 && st0.b.offset == 0 && st0.b.stride == 1)

    val m = st.length
    val n = st0.a.size
    val k = st0.b.size
    
    val AData = new Array[Double](m * n)
    var i = 0
    while (i < m) {
      System.arraycopy(st(i).a.data, 0, AData, i * n, n)
      i += 1
    }
    
    val BData = new Array[Double](k * m)
    var q = 0.0
    i = 0
    while (i < m) {
      q += st(i).q
      System.arraycopy(st(i).b.data, 0, BData, i * k, k)
      i += 1
    }

    val A = new DenseMatrix[Double](n, m, AData)
    val B = new DenseMatrix[Double](k, m, BData)
    
    (q, A.t * A - B.t * B)
  }

  def computeGramianLargeN(st: Array[SkatTuple]): (Double, DenseMatrix[Double]) = {
    require(st.nonEmpty)
    
    val m = st.length
    val data = Array.ofDim[Double](m * m)
    var q = 0.0

    var i = 0
    while (i < m) {
      q += st(i).q
      val ai = st(i).a
      val bi = st(i).b
      data(i * (m + 1)) = (ai dot ai) - (bi dot bi)
      var j = 0
      while (j < i) {
        val temp = (ai dot st(j).a) - (bi dot st(j).b)
        data(i * m + j) = temp
        data(j * m + i) = temp
        j += 1
      }
      i += 1
    }
    
    (q, new DenseMatrix[Double](m, m, data))
  }
  
  /** Davies Algorithm original C code
  *   Citation:
  *     Davies, Robert B. "The distribution of a linear combination of
  *     x2 random variables." Applied Statistics 29.3 (1980): 323-333.
  *   Software Link:
  *     http://www.robertnz.net/QF.htm
  */
  @native
  def qfWrapper(lb1: Array[Double], nc1: Array[Double], n1: Array[Int], r1: Int,
    sigma: Double, c1: Double, lim1: Int, acc: Double, trace: Array[Double],
    ifault: IntByReference): Double

  Native.register("hail")
  
  // gramian is the m x m matrix (G * sqrt(W)).t * P_0 * (G * sqrt(W)) which has the same non-zero eigenvalues
  // as the n x n matrix in the paper P_0^{1/2} * (G * W * G.t) * P_0^{1/2}
  def computePval(q: Double, gramian: DenseMatrix[Double], accuracy: Double, iterations: Int): (Double, Int) = {
    val allEvals = eigSymD.justEigenvalues(gramian)

    // filter out those eigenvalues below the mean / 100k
    val threshold = 1e-5 * sum(allEvals) / allEvals.length
    val evals = allEvals.toArray.dropWhile(_ < threshold) // evals are increasing

    val terms = evals.length
    val noncentrality = Array.fill[Double](terms)(0.0)
    val dof = Array.fill[Int](terms)(1)
    val trace = Array.fill[Double](7)(0.0)
    val fault = new IntByReference()
    val s = 0.0
    val x = qfWrapper(evals, noncentrality, dof, terms, s, q, iterations, accuracy, trace, fault)
    val pval = 1 - x
    
    (pval, fault.getValue)
  }
}

