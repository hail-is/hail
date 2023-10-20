package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._
import is.hail.HailContext
import is.hail.annotations.{Annotation, Region, UnsafeRow}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.stats.{GeneralizedChiSquaredDistribution, LogisticRegressionModel, RegressionUtils, eigSymD}
import is.hail.types._
import is.hail.types.virtual.{TFloat64, TInt32, TStruct, Type}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

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
case class SkatTuple(q: Double, a: BDV[Double], b: BDV[Double])

object Skat {
  def computeGramianSmallN(st: Array[SkatTuple]): (Double, BDM[Double]) = {
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

    val A = new BDM[Double](n, m, AData)
    val B = new BDM[Double](k, m, BData)

    (q, A.t * A - B.t * B)
  }

  def computeGramianLargeN(st: Array[SkatTuple]): (Double, BDM[Double]) = {
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

    (q, new BDM[Double](m, m, data))
  }

  def computeGramian(st: Array[SkatTuple], useSmallN: Boolean): (Double, BDM[Double]) =
    if (useSmallN) computeGramianSmallN(st) else computeGramianLargeN(st)

  // gramian is the m x m matrix (G * sqrt(W)).t * P_0 * (G * sqrt(W)) which has the same non-zero eigenvalues
  // as the n x n matrix in the paper P_0^{1/2} * (G * W * G.t) * P_0^{1/2}
  def computePval(q: Double, gramian: BDM[Double], accuracy: Double, iterations: Int): (Double, Int) = {
    val allEvals = eigSymD.justEigenvalues(gramian)

    // filter out those eigenvalues below the mean / 100k
    val threshold = 1e-5 * sum(allEvals) / allEvals.length
    val evals = allEvals.toArray.dropWhile(_ < threshold) // evals are increasing

    val terms = evals.length
    val noncentrality = Array.fill[Double](terms)(0.0)
    val dof = Array.fill[Int](terms)(1)
    val s = 0.0

    val result = GeneralizedChiSquaredDistribution.cdfReturnExceptions(
      q, dof, evals, noncentrality, s, iterations, accuracy
    )
    val x = result.value
    val nIntegrations = result.nIterations
    val converged = result.converged
    val fault = result.fault
    val pval = 1 - x

    (pval, fault)
  }
}

case class Skat(
  keyField: String,
  weightField: String,
  yField: String,
  xField: String,
  covFields: Seq[String],
  logistic: Boolean,
  maxSize: Int,
  accuracy: Double,
  iterations: Int,
  logistic_max_iterations: Int,
  logistic_tolerance: Double
) extends MatrixToTableFunction {

  assert(logistic || logistic_max_iterations == 0 && logistic_tolerance == 0.0)

  val hardMaxEntriesForSmallN = 64e6 // 8000 x 8000 => 512MB of doubles

  override def typ(childType: MatrixType): TableType = {
    val keyType = childType.rowType.fieldType(keyField)
    val skatSchema = TStruct(
      ("id", keyType),
      ("size", TInt32),
      ("q_stat", TFloat64),
      ("p_value", TFloat64),
      ("fault", TInt32))
    TableType(skatSchema, FastSeq("id"), TStruct.empty)
  }

  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {

    if (maxSize <= 0 || maxSize > 46340)
      fatal(s"Maximum group size must be in [1, 46340], got $maxSize")

    val maxEntriesForSmallN = math.min(maxSize * maxSize, hardMaxEntriesForSmallN)

    if (accuracy <= 0)
      fatal(s"tolerance must be positive, default is 1e-6, got $accuracy")
    if (iterations <= 0)
      fatal(s"iterations must be positive, default is 10000, got $iterations")

    val (y, cov, completeColIdx) = RegressionUtils.getPhenoCovCompleteSamples(mv, yField, covFields.toArray)

    val n = y.size
    val k = cov.cols
    val d = n - k

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } (including intercept) implies $d degrees of freedom.")
    if (logistic) {
      val badVals = y.findAll(yi => yi != 0d && yi != 1d)
      if (badVals.nonEmpty)
        fatal(s"For logistic SKAT, phenotype must be Boolean or numeric with value 0 or 1 for each complete " +
          s"sample; found ${badVals.length} ${plural(badVals.length, "violation")} starting with ${badVals(0)}")
    }

    val (keyGsWeightRdd, keyType) =
      computeKeyGsWeightRdd(mv, xField, completeColIdx, keyField, weightField)

    val backend = HailContext.backend

    def linearSkat(): RDD[Row] = {
      // fit null model
      val (qt, res) =
        if (k == 0)
          (BDM.zeros[Double](0, n), y)
        else {
          val QR = qr.reduced(cov)
          val Qt = QR.q.t
          val R = QR.r
          val beta = R \ (Qt * y)
          (Qt, y - cov * beta)
        }
      val sigmaSq = (res dot res) / d

      val resBc = backend.broadcast(res)
      val QtBc = backend.broadcast(qt)

      def linearTuple(x: BDV[Double], w: Double): SkatTuple = {
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
            val (q, gramian) = Skat.computeGramian(skatTuples, size.toLong * n <= maxEntriesForSmallN)

            // using q / sigmaSq since Z.t * Z = gramian / sigmaSq
            val (pval, fault) = Skat.computePval(q / sigmaSq, gramian, accuracy, iterations)

            // returning qstat = q / (2 * sigmaSq) to agree with skat R table convention
            Row(key, size, q / (2 * sigmaSq), pval, fault)
          } else {
            Row(key, size, null, null, null)
          }
        }
    }

    def logisticSkat(): RDD[Row] = {
      val (sqrtV, res, cinvXtV) =
        if (k > 0) {
          val logRegM = new LogisticRegressionModel(cov, y).fit(maxIter=logistic_max_iterations, tol=logistic_tolerance)
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
          var Cinv: BDM[Double] = null
          try {
            Cinv = inv(cholesky(XtVX))
          } catch {
            case e: MatrixSingularException =>
              fatal("Singular matrix exception while computing Cholesky factor of X.t * V * X.\n" + e.getMessage)
            case e: NotConvergedException =>
              fatal("Not converged exception while inverting Cholesky factor of X.t * V * X.\n" + e.getMessage)
          }
          (sqrt(V), y - mu, Cinv * VX.t)
        } else
          (BDV.fill(n)(0.5), y, new BDM[Double](0, n))

      val sqrtVBc = backend.broadcast(sqrtV)
      val resBc = backend.broadcast(res)
      val CinvXtVBc = backend.broadcast(cinvXtV)

      def logisticTuple(x: BDV[Double], w: Double): SkatTuple = {
        val xw = x * math.sqrt(w)
        val sqrt_q = resBc.value dot xw
        SkatTuple(sqrt_q * sqrt_q, xw *:* sqrtVBc.value , CinvXtVBc.value * xw)
      }

      keyGsWeightRdd.map { case (key, vs) =>
        val vsArray = vs.toArray
        val size = vsArray.length
        if (size <= maxSize) {
          val skatTuples = vs.map((logisticTuple _).tupled).toArray
          val (q, gramian) = Skat.computeGramian(skatTuples, size.toLong * n <= maxEntriesForSmallN)
          val (pval, fault) = Skat.computePval(q, gramian, accuracy, iterations)

          // returning qstat = q / 2 to agree with skat R table convention
          Row(key, size, q / 2, pval, fault)
        } else {
          Row(key, size, null, null, null)
        }
      }.persist()
    }

    val skatRdd = if (logistic) logisticSkat() else linearSkat()

    val tableType = typ(mv.typ)
    TableValue(ctx, tableType.rowType, tableType.key, skatRdd)
  }

  def computeKeyGsWeightRdd(mv: MatrixValue,
    xField: String,
    completeColIdx: Array[Int],
    keyField: String,
    // returns ((key, [(gs_v, weight_v)]), keyType)
    weightField: String): (RDD[(Annotation, Iterable[(BDV[Double], Double)])], Type) = {

    val fullRowType = mv.rvRowPType
    val keyStructField = fullRowType.field(keyField)
    val keyIndex = keyStructField.index
    val keyType = keyStructField.typ

    val weightStructField = fullRowType.field(weightField)
    val weightIndex = weightStructField.index
    assert(weightStructField.typ.virtualType == TFloat64)

    val entryArrayType = mv.entryArrayPType
    val entryType = mv.entryPType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.virtualType == TFloat64)

    val entryArrayIdx = mv.entriesIdx
    val fieldIdx = entryType.fieldIdx(xField)

    val n = completeColIdx.length
    val completeColIdxBc = HailContext.backend.broadcast(completeColIdx)

    // I believe no `boundary` is needed here because `mapPartitions` calls `run` which calls `cleanupRegions`.
    (mv.rvd.mapPartitions { (ctx, it) => it.flatMap { ptr =>
      val keyIsDefined = fullRowType.isFieldDefined(ptr, keyIndex)
      val weightIsDefined = fullRowType.isFieldDefined(ptr, weightIndex)

      if (keyIsDefined && weightIsDefined) {
        val weight = Region.loadDouble(fullRowType.loadField(ptr, weightIndex))
        if (weight < 0)
          fatal(s"Row weights must be non-negative, got $weight")
        val key = Annotation.copy(keyType.virtualType, UnsafeRow.read(keyType, fullRowType.loadField(ptr, keyIndex)))
        val data = new Array[Double](n)

        RegressionUtils.setMeanImputedDoubles(data, 0, completeColIdxBc.value, new IntArrayBuilder(),
          ptr, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)
        Some(key -> (BDV(data) -> weight))
      } else None
    }
    }.groupByKey(), keyType.virtualType)
  }

}
