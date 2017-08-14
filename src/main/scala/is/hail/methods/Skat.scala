package is.hail.methods

import is.hail.utils._
import is.hail.variant._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats.{RegressionUtils, SkatModel}
import is.hail.annotations.Annotation
import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object SkatStat {
  val schema = TStruct(
    ("q", TFloat64),
    ("pval", TFloat64),
    ("fault", TInt32))
}

case class SkatStat(q: Double, pval: Double, fault: Int) {
  def toAnnotation = Annotation(q, pval, fault)
}

case class SkatTuple[T <: Vector[Double]](q: Double, xw: T, qtxw: DenseVector[Double])

object Skat {

  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    yExpr: String,
    covExpr: Array[String],
    useDosages: Boolean,
    useLargeN: Boolean = false): KeyTable = {


    def SkatRDDtoKeyTable[T <: Vector[Double]](keyedRdd:  RDD[(Any, Iterable[(T, Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double], keyName: String,
      resultOp: (Array[SkatTuple[T]], Double) => SkatStat): KeyTable = {
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

      def variantPreProcess[T <: Vector[Double]](gs: T, w: Double): SkatTuple[T] = {
        val sqrtw = math.sqrt(w.asInstanceOf[Double])
        val wx: T = (gs * sqrtw).asInstanceOf[T]
        val sj = resBc.value dot wx
        SkatTuple(sj * sj, wx, qtBc.value * wx)
      }

      val skatRDD = keyedRdd
        .map {case (k, vs) =>
          val vArray = vs.toArray.map{ case (gs, w) => variantPreProcess(gs, w) }
          val skatStat = if (vArray.length * n < Int.MaxValue) {
            resultOp(vArray, sigmaSq)
          }
          else {
            largeNResultOp(vArray, sigmaSq)
          }
          Row(k, skatStat.q, skatStat.pval, skatStat.fault)
        }

      val (skatSignature, _) = TStruct(keyName -> keyType.asInstanceOf[Type]).merge(SkatStat.schema)

      new KeyTable(vds.hc, skatRDD, skatSignature, key = Array(keyName))
    }


    val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)

    (useDosages,useLargeN) match {
      case (false, false) =>
        val (keyedRdd, keysType, y, cov) =
          keyedRDDSkat(vds, variantKeys, singleKey, weightExpr, yExpr, covExpr, RegressionUtils.hardCalls(_, _))
        SkatRDDtoKeyTable(keyedRdd, keysType, y, cov, keyName, sparseResultOp)
      case (false, true) =>
        val (keyedRdd, keysType, y, cov) =
          keyedRDDSkat(vds, variantKeys, singleKey, weightExpr, yExpr, covExpr, RegressionUtils.hardCalls(_, _))
        SkatRDDtoKeyTable(keyedRdd, keysType, y, cov, keyName, largeNResultOp[SparseVector[Double]])
      case (true, false) =>
        val (keyedRdd, keysType, y, cov) =
          keyedRDDSkat(vds, variantKeys, singleKey, weightExpr, yExpr, covExpr, dosages)
        SkatRDDtoKeyTable(keyedRdd, keysType, y, cov, keyName, denseResultOp)
      case (true, true) =>
        val (keyedRdd, keysType, y, cov) =
          keyedRDDSkat(vds, variantKeys, singleKey, weightExpr, yExpr, covExpr, dosages)
        SkatRDDtoKeyTable(keyedRdd, keysType, y, cov, keyName, largeNResultOp[DenseVector[Double]])
    }

  }

  def denseResultOp(st: Array[SkatTuple[DenseVector[Double]]], sigmaSq: Double): SkatStat = {

    val m = st.length
    val n = st(0).xw.size
    val k = st(0).qtxw.size

    var xwArray = new Array[Double](m * n)
    var qtxwArray = new Array[Double](m * k)
    var j = 0;
    var i = 0

    //copy in non-zeros to weighted genotype matrix array
    i = 0
    while (i < m) {
      j = 0
      val xwsi = st(i).xw
      while (j < n) {
        xwArray(i * n + j) = xwsi(j)
        j += 1
      }
      i += 1
    }

    //add in non-zeros to QtGW array
    i = 0
    while (i < m) {
      j = 0
      val qtxwsi = st(i).qtxw
      while (j < k) {
        qtxwArray(i * k + j) = qtxwsi.data(j)
        j += 1
      }
      i += 1
    }

    //compute the variance component score
    var skatStat = 0.0
    i = 0
    while (i < m) {
      skatStat += st(i).q
      i += 1
    }

    val weightedGenotypes = new DenseMatrix[Double](n, m, xwArray)
    val qtWeightedGenotypes = new DenseMatrix[Double](k, m, qtxwArray)
    val GwGrammian = weightedGenotypes.t * weightedGenotypes
    val QtGwGrammian = qtWeightedGenotypes.t * qtWeightedGenotypes

    val SPG = new SkatModel(skatStat / (2 * sigmaSq))
    SPG.computeLinearSkatStats(GwGrammian, QtGwGrammian)
  }

  def sparseResultOp(st: Array[SkatTuple[SparseVector[Double]]], sigmaSq: Double): SkatStat = {

    val m = st.length
    val n = st(0).xw.size
    val k = st(0).qtxw.size

    val xwArray = new Array[Double](m * n)
    val qtxwArray = new Array[Double](m * k)

    var j = 0;
    var i = 0

    while (i < m) {
      val nnz = st(i).xw.used
      val xwsi = st(i).xw
      j = 0
      while (j < nnz) {
        val index = st(i).xw.index(j)
        xwArray(i * n + index) = xwsi.data(j)
        j += 1
      }
      i += 1
    }

    //add in non-zeros to QtGW array
    i = 0
    while (i < m) {
      j = 0
      val qtxwsi = st(i).qtxw
      while (j < k) {
        qtxwArray(i * k + j) = qtxwsi.data(j)
        j += 1
      }
      i += 1
    }

    //compute the variance component score
    var skatStat = 0.0
    i = 0
    while (i < m) {
      skatStat += st(i).q
      i += 1
    }

    val weightedGenotypes = new DenseMatrix[Double](n, m, xwArray)
    val qtWeightedGenotypes = new DenseMatrix[Double](k, m, qtxwArray)
    val GwGrammian = weightedGenotypes.t * weightedGenotypes
    val QtGwGrammian = qtWeightedGenotypes.t * qtWeightedGenotypes

    val SPG = new SkatModel(skatStat / (2 * sigmaSq))
    SPG.computeLinearSkatStats(GwGrammian, QtGwGrammian)
  }

  def largeNResultOp[T <: Vector[Double]](st: Array[SkatTuple[T]], sigmaSq: Double): SkatStat = {
    val m = st.length
    val n = st(0).xw.size
    val k = st(0).qtxw.size

    val ZGrammianArray = new Array[Double](m * m)
    val QtZGrammianArray = new Array[Double](m * m)

    var i = 0
    var j = 0

    while (i < m) {
      ZGrammianArray(i * m + i) = st(i).xw dot st(i).xw
      j = 0
      while (j < i) {
        val ijdotprod = st(i).xw dot st(j).xw
        ZGrammianArray(i * m + j) = ijdotprod
        ZGrammianArray(j * m + i) = ijdotprod
        j += 1
      }
      i += 1
    }

    i = 0
    while (i < m) {
      QtZGrammianArray(i * m + i) = st(i).qtxw dot st(i).qtxw
      j = 0
      while (j < i) {
        val ijdotprod = st(i).qtxw dot st(j).qtxw
        QtZGrammianArray(i * m + j) = ijdotprod
        QtZGrammianArray(j * m + i) = ijdotprod
        j += 1
      }
      i += 1
    }

    //compute the variance component score
    var skatStat = 0.0
    i = 0
    while (i < m) {
      skatStat += st(i).q
      i += 1
    }

    val weightedGenotypesGrammian = new DenseMatrix[Double](m, m, ZGrammianArray)
    val qtWeightedGenotypesGrammian = new DenseMatrix[Double](m, m, QtZGrammianArray)

    val SPG = new SkatModel(skatStat / (2 * sigmaSq))
    SPG.computeLinearSkatStats(weightedGenotypesGrammian, qtWeightedGenotypesGrammian)
  }

  def keyedRDDSkat[T <: Vector[Double]](vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    yExpr: String,
    covExpr: Array[String],
    getGenotypes: (Iterable[Genotype], Int) => T):
  (RDD[(Any, Iterable[(T, Double)])], Type, DenseVector[Double], DenseMatrix[Double]) = {

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)

    val n = y.size

    val filteredVds = weightExpr match {
      case None => vds.filterSamplesList(completeSamples.toSet)
        .annotateVariantsExpr("va.AF = gs.callStats(g=> v).AF")
        .annotateVariantsExpr("va.weight = let af = if (va.AF[0] <= va.AF[1]) va.AF[0] else va.AF[1] in dbeta(af,1.0,25.0)**2")
      case _ => vds.filterSamplesList(completeSamples.toSet)
    }

    val (keysType, keysQuerier) = filteredVds.queryVA(variantKeys)
    val (weightType, weightQuerier) = weightExpr match {
      case None => filteredVds.queryVA("va.weight")
      case Some(expr) => filteredVds.queryVA(expr)
    }

    val typedWeightQuerier = weightType match {
      case TFloat64 => weightQuerier.asInstanceOf[Annotation => Double]
      case TFloat32 => (x: Annotation) => weightQuerier(x).asInstanceOf[Float].toDouble
      case TInt64 => (x: Annotation) => weightQuerier(x).asInstanceOf[Long].toDouble
      case TInt32 => (x: Annotation) => weightQuerier(x).asInstanceOf[Int].toDouble
      case _ => fatal("Weight must evaluate to numeric type")
    }

    val (keyType, keyIterator): (Any, Any => Iterator[Any]) = if (singleKey) {
      (keysType, (key: Any) => Iterator.single(key))
    } else {
      val keyType = keysType match {
        case t: TIterable => t.elementType
        case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
      }
      (keyType, (keys: Any) => keys.asInstanceOf[Iterable[Any]].iterator)
    }

    (filteredVds.rdd.flatMap { case (v, (va, gs)) =>
      (Option(keysQuerier(va)), Option(typedWeightQuerier(va))) match {
        case (Some(key), Some(w)) =>
          val gVector = getGenotypes(gs, n)
          keyIterator(key).map((_, gVector -> w))
        case _ =>
          Iterator.empty
      }
    }.groupByKey(),keysType, y, cov)

  }
}
