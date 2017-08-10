package is.hail.methods

import is.hail.utils._
import is.hail.variant._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats.{RegressionUtils, SkatModel}
import is.hail.annotations.Annotation
import breeze.linalg._
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

object SkatAgg {
  val zeroValDense = SkatAgg(0.0, new ArrayBuilder[DenseVector[Double]](), new ArrayBuilder[DenseVector[Double]]())
  val zeroValSparse = SkatAgg(0.0, new ArrayBuilder[SparseVector[Double]](), new ArrayBuilder[DenseVector[Double]]())

  def seqOp[T <: Vector[Double]](sa: SkatAgg[T], st: SkatTuple[T]): SkatAgg[T] =
    SkatAgg[T](sa.q + st.q, sa.xws + st.xw, sa.qtxws + st.qtxw)

  def combOp[T <: Vector[Double]](sa: SkatAgg[T], sa2: SkatAgg[T]): SkatAgg[T] =
    SkatAgg[T](sa.q + sa2.q, sa.xws ++ sa2.xws.result(), sa.qtxws ++ sa2.qtxws.result())

  def denseResultOp(sa: SkatAgg[DenseVector[Double]], sigmaSq: Double): SkatStat = {

    val m = sa.xws.length
    val n = sa.xws(0).size
    val k = sa.qtxws(0).length

    var xwArray = new Array[Double](m * n)
    var qtxwArray = new Array[Double](m * k)
    var j = 0;
    var i = 0

    //copy in non-zeros to weighted genotype matrix array
    i = 0
    while (i < m) {
      j = 0
      val xwsi = sa.xws(i)
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
      val qtxwsi = sa.qtxws(i)
      while (j < k) {
        qtxwArray(i * k + j) = qtxwsi.data(j)
        j += 1
      }
      i += 1
    }

    val weightedGenotypes = new DenseMatrix[Double](n, m, xwArray)
    val qtWeightedGenotypes = new DenseMatrix[Double](k, m, qtxwArray)
    val GwGrammian = weightedGenotypes.t * weightedGenotypes
    val QtGwGrammian = qtWeightedGenotypes.t * qtWeightedGenotypes

    val SPG = new SkatModel(sa.q / (2 * sigmaSq))
    SPG.computeLinearSkatStats(GwGrammian, QtGwGrammian)
  }

  def sparseResultOp(sa: SkatAgg[SparseVector[Double]], sigmaSq: Double): SkatStat = {

    val m = sa.xws.length
    val n = sa.xws(0).size
    val k = sa.qtxws(0).length

    val xwArray = new Array[Double](m * n)
    val qtxwArray = new Array[Double](m * k)

    var j = 0;
    var i = 0

    while (i < m) {
      val nnz = sa.xws(i).used
      val xwsi = sa.xws(i)
      j = 0
      while (j < nnz) {
        val index = sa.xws(i).index(j)
        xwArray(i * n + index) = xwsi.data(j)
        j += 1
      }
      i += 1
    }

    //add in non-zeros to QtGW array
    i = 0
    while (i < m) {
      j = 0
      val qtxwsi = sa.qtxws(i)
      while (j < k) {
        qtxwArray(i * k + j) = qtxwsi.data(j)
        j += 1
      }
      i += 1
    }

    val weightedGenotypes = new DenseMatrix[Double](n, m, xwArray)
    val qtWeightedGenotypes = new DenseMatrix[Double](k, m, qtxwArray)
    val GwGrammian = weightedGenotypes.t * weightedGenotypes
    val QtGwGrammian = qtWeightedGenotypes.t * qtWeightedGenotypes

    val SPG = new SkatModel(sa.q / (2 * sigmaSq))
    SPG.computeLinearSkatStats(GwGrammian, QtGwGrammian)
  }

  def largeNResultOp[T <: Vector[Double]](sa: SkatAgg[T], sigmaSq: Double): SkatStat = {
    val m = sa.xws.length
    val n = sa.xws(0).size
    val k = sa.qtxws(0).length

    val ZGrammianArray = new Array[Double](m * m)
    val QtZGrammianArray = new Array[Double](m * m)

    var i = 0
    var j = 0

    while (i < m) {
      ZGrammianArray(i * m + i) = sa.xws(i) dot sa.xws(i)
      j = 0
      while (j < i) {
        val ijdotprod = sa.xws(i) dot sa.xws(j)
        ZGrammianArray(i * m + j) = ijdotprod
        ZGrammianArray(j * m + i) = ijdotprod
        j += 1
      }
      i += 1
    }

    i = 0
    while (i < m) {
      QtZGrammianArray(i * m + i) = sa.qtxws(i) dot sa.qtxws(i)
      j = 0
      while (j < i) {
        val ijdotprod = sa.qtxws(i) dot sa.qtxws(j)
        QtZGrammianArray(i * m + j) = ijdotprod
        QtZGrammianArray(j * m + i) = ijdotprod
        j += 1
      }
      i += 1
    }
    val weightedGenotypesGrammian = new DenseMatrix[Double](m, m, ZGrammianArray)
    val qtWeightedGenotypesGrammian = new DenseMatrix[Double](m, m, QtZGrammianArray)

    val SPG = new SkatModel(sa.q / (2 * sigmaSq))
    SPG.computeLinearSkatStats(weightedGenotypesGrammian, qtWeightedGenotypesGrammian)
  }

}

case class SkatAgg[T <: Vector[Double]](q: Double, xws: ArrayBuilder[T], qtxws: ArrayBuilder[DenseVector[Double]])

object Skat {
  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    yExpr: String,
    covExpr: Array[String],
    use_dosages: Boolean): KeyTable = {

    if (!use_dosages)
      polymorphicSkat(vds, keyName, variantKeys, singleKey, weightExpr, yExpr, covExpr,
        RegressionUtils.hardCalls(_, _), SkatAgg.zeroValSparse, SkatAgg.sparseResultOp _)
    else {
      val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)
      polymorphicSkat(vds, keyName, variantKeys, singleKey, weightExpr, yExpr, covExpr,
        dosages, SkatAgg.zeroValDense, SkatAgg.denseResultOp _)
    }
  }

  def polymorphicSkat[T <: Vector[Double]](vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    yExpr: String,
    covExpr: Array[String],
    getGenotypes: (Iterable[Genotype], Int) => T,
    zero: SkatAgg[T],
    resultOp: (SkatAgg[T], Double) => SkatStat): KeyTable = {

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)

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

    //ask Tim about how to resolve this bug
    val typedWeightQuerier = weightType match {
      case TFloat64 => weightQuerier.asInstanceOf[Annotation => Double]
      case TFloat32 => weightQuerier.asInstanceOf[Annotation => Double]
      case TInt64 => weightQuerier.asInstanceOf[Annotation => Double]
      case TInt32 => weightQuerier.asInstanceOf[Annotation => Double]
      case _ => fatal("Weight must evaluate to numeric type")
    }

    val sc = filteredVds.sparkContext
    val resBc = sc.broadcast(res)

    def variantPreProcess(gs: Iterable[Genotype], w: Double): SkatTuple[T] = {
      val sqrtw = math.sqrt(w.asInstanceOf[Double])
      val wx: T = (getGenotypes(gs, n) * sqrtw).asInstanceOf[T]
      val sj = resBc.value dot wx
      SkatTuple(sj * sj, wx, q.t * wx)
    }

    val (keyType, keyIterator): (Any, Any => Iterator[Any]) = if (singleKey) {
      (keysType, (key: Any) => Iterator.single(key))
    } else {
      val keyType = keysType match {
        case TArray(e) => e
        case TSet(e) => e
        case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
      }
      (keyType, (keys: Any) => keys.asInstanceOf[Iterable[Any]].iterator)
    }

    val keyedRdd = filteredVds.rdd.flatMap { case (v, (va, gs)) =>
      (Option(keysQuerier(va)), Option(typedWeightQuerier(va))) match {
        case (Some(key), Some(w)) =>
          keyIterator(key).map((_, variantPreProcess(gs, w.asInstanceOf[Double])))
        case _ =>
          Iterator.empty
      }
    }

    val aggregatedKT = keyedRdd.aggregateByKey(zero)(SkatAgg.seqOp, SkatAgg.combOp)

    val skatRDD = aggregatedKT.map { case (key, value) =>
      val skatStat = if (value.xws.length * n < Int.MaxValue) {
        resultOp(value, sigmaSq)
      }
      else {
        SkatAgg.largeNResultOp(value, sigmaSq)
      }

      Row(key, skatStat.q, skatStat.pval, skatStat.fault)
    }

    val (skatSignature, _) = TStruct(keyName -> keyType.asInstanceOf[Type]).merge(SkatStat.schema)

    new KeyTable(filteredVds.hc, skatRDD, skatSignature, key = Array(keyName))
  }
}
