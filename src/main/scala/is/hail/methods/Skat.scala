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

object SkatStat {
  def schema(keyType: Type) = TStruct(
    ("key", keyType),
    ("qstat", TFloat64),
    ("pval", TFloat64),
    ("fault", TInt32))
}

case class SkatStat(q: Double, p: Double, fault: Int)

case class SkatTuple(q: Double, xw: Vector[Double], qtxw: DenseVector[Double])

object Skat {
  def apply(vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    yExpr: String,
    covExpr: Array[String],
    weightExpr: Option[String],
    useLogistic: Boolean = false,
    useDosages: Boolean,
    useLargeN: Boolean = false): KeyTable = {
    val (y, cov, completeSampleIndex) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)

    if (useLogistic && (!y.forall(yi => yi == 0d || yi == 1d)))
      fatal(s"For logistic SKAT, phenotype must be Boolean or numeric with all present values equal to 0 or 1")

    val n = y.size
    val sampleMask = Array.fill[Boolean](vds.nSamples)(false)
    completeSampleIndex.foreach(i => sampleMask(i) = true)
    val filteredVds = vds.filterSamplesMask(sampleMask)

    def computeSkatLinear(keyedRdd: RDD[(Any, Iterable[(Vector[Double], Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double],
      resultOp: (Array[SkatTuple], Double) => SkatStat): KeyTable = {
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
      val qtBc = sc.broadcast(q.t)

      def preProcessGenotypes(gs: Vector[Double], w: Double): SkatTuple = {
        if (w < 0)
          fatal(s"Variant weights must be non-negative, got $w")
        val sqrtw = math.sqrt(w)
        val wx = gs * sqrtw
        val sj = resBc.value dot wx
        SkatTuple(sj * sj, wx, qtBc.value * wx)
      }

      val skatRDD = keyedRdd
        .map { case (key, vs) =>
          val vArray = vs.toArray.map { case (gs, w) => preProcessGenotypes(gs, w) }
          val skatStat = if (vArray.length.toLong * n < Int.MaxValue) {
            resultOp(vArray, 2 * sigmaSq)
          } else {
            runSkatPerKeyLargeN(vArray, 2 * sigmaSq)
          }
          Row(key, skatStat.q, skatStat.p, skatStat.fault)
        }

      val skatSignature = SkatStat.schema(keyType)

      new KeyTable(vds.hc, skatRDD, skatSignature, Array("key"))
    }

    def computeSkatLogistic(keyedRdd: RDD[(Any, Iterable[(Vector[Double], Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double],
      resultOp: (Array[SkatTuple], Double) => SkatStat): KeyTable = {
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
      val XVX = cov.t * VX
      XVX.forceSymmetry()
      val Cinv = inv(cholesky(XVX))
      val res = y - mu

      val sc = keyedRdd.sparkContext
      val sqrtVBc = sc.broadcast(sqrt(V))
      val resBc = sc.broadcast(res)
      val CinvXtVBc = sc.broadcast(Cinv * VX.t)

      def variantPreProcess(gs: Vector[Double], w: Double): SkatTuple = {
        val wx = gs * math.sqrt(w)
        val sj = resBc.value dot wx
        val CinvXtVwx = CinvXtVBc.value * wx
        SkatTuple(sj * sj, wx :* sqrtVBc.value , CinvXtVwx)
      }

      val skatRDD = keyedRdd
        .map { case (key, vs) =>
          val vArray = vs.toArray.map { case (gs, w) => variantPreProcess(gs, w) }
          val skatStat = resultOp(vArray, 2.0)
          Row(key, skatStat.q, skatStat.p, skatStat.fault)
        }

      val skatSignature = SkatStat.schema(keyType)

      new KeyTable(vds.hc, skatRDD, skatSignature, Array("key"))
    }

    val completeSamplesBc = filteredVds.sparkContext.broadcast((0 until n).toArray)

    val getGenotypesFunction = (gs: Iterable[Genotype], n: Int) =>
      if (!useDosages) {
        RegressionUtils.hardCalls(gs, n)
      } else {
        RegressionUtils.dosages(gs, completeSamplesBc.value)
      }

    val (keyedRdd, keysType) = keyedRDDSkat(filteredVds, variantKeys, singleKey, weightExpr, getGenotypesFunction)

    if (!useLogistic) computeSkatLinear(keyedRdd, keysType, y, cov, if (!useLargeN) runSkatPerKey else runSkatPerKeyLargeN)
    else computeSkatLogistic(keyedRdd, keysType, y, cov, if (!useLargeN) runSkatPerKey else runSkatPerKeyLargeN)
  }

  def keyedRDDSkat(vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    getGenotypes: (Iterable[Genotype], Int) => Vector[Double]):
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

    (vdsWithWeight.rdd.flatMap { case (v, (va, gs)) =>
      (Option(keysQuerier(va)), Option(typedWeightQuerier(va))) match {
        case (Some(key), Some(w)) =>
          val gVector = getGenotypes(gs, n)
          keyIterator(key).map((_, gVector -> w))
        case _ => Iterator.empty
      }
    }.groupByKey(), keysType)
  }

  /*
  Davies algorithm takes the eigenvalues of (G * sqrt(W)).t * P_0 * G * sqrt(W)
  We evaluate this matrix as 0.5 * (A.t * A - B.t * B) where
  linear:   A = G * sqrt(W)              B = Q.t * G * sqrt(W)
  logistic: A = sqrt(V) * G * sqrt(W)    B = inv(L) * X.t * V * G * sqrt(W)
  Here X = Q * R and L * L.t = X.t * V * X
  */
  def runSkatPerKey(st: Array[SkatTuple], skatStatScaling: Double): SkatStat = {
    require(st.nonEmpty)
    
    val m = st.length
    val n = st(0).xw.size    
    val k = st(0).qtxw.size
    val isDenseVector = st(0).xw.isInstanceOf[DenseVector[Double]]
        
    var AArray = new Array[Double](m * n)

    var i = 0
    if (isDenseVector) {
      while (i < m) {
        val xwsi = st(i).xw
        var j = 0
        while (j < n) {
          AArray(i * n + j) = xwsi(j)
          j += 1
        }
        i += 1
      }
    } else {
      while (i < m) {
        val xwsi = st(i).xw.asInstanceOf[SparseVector[Double]]
        val nnz = xwsi.used
        var j = 0
        while (j < nnz) {
          val index = xwsi.index(j)
          AArray(i * n + index) = xwsi.data(j)
          j += 1
        }
        i += 1
      }
    }
    
    var BArray = new Array[Double](k * m)
      
    i = 0
    while (i < m) {
      var j = 0
      val CinvXtVZi = st(i).qtxw
      while (j < k) {
        BArray(i * k + j) = CinvXtVZi(j)
        j += 1
      }
      i += 1
    }
    
    var skatStat = 0.0
    i = 0
    while (i < m) {
      skatStat += st(i).q
      i += 1
    }

    val A = new DenseMatrix[Double](n, m, AArray)
    val B = new DenseMatrix[Double](k, m, BArray)

    val model = new SkatModel(skatStat / skatStatScaling)
    model.computePVal(0.5 * (A.t * A - B.t * B))
  }

  def runSkatPerKeyLargeN(st: Array[SkatTuple], skatStatScaling: Double): SkatStat = {
    val m = st.length

    // directly compute the entries of D = 0.5 * (A.t * A - B.t * B)
    val DArray = Array.ofDim[Double](m * m)
    
    var i = 0
    while (i < m) {
      val xwsi = st(i).xw
      val qtxwsi = st(i).qtxw
      DArray(i * (m + 1)) = 0.5 * ((xwsi dot xwsi) - (qtxwsi dot qtxwsi))
      var j = 0
      while (j < i) {
        val temp = 0.5 * ((xwsi dot st(j).xw) - (qtxwsi dot st(j).qtxw))
        DArray(i * m + j) = temp
        DArray(j * m + i) = temp
        j += 1
      }
      i += 1
    }

    var skatStat = 0.0
    i = 0
    while (i < m) {
      skatStat += st(i).q
      i += 1
    }

    val model = new SkatModel(skatStat / skatStatScaling)
    model.computePVal(new DenseMatrix[Double](m, m, DArray))
  }
}
