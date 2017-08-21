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
  def schema(keyType: Type) = TStruct(
    ("key", keyType),
    ("q", TFloat64),
    ("pval", TFloat64),
    ("fault", TInt32))
}

case class SkatStat(q: Double, pval: Double, fault: Int)

case class SkatTuple(q: Double, xw: Vector[Double], qtxw: DenseVector[Double])

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
    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    var filteredVds = vds.filterSamplesList(completeSamples.toSet)
    val n = y.size

    def computeSkat(keyedRdd: RDD[(Any, Iterable[(Vector[Double], Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double], keyName: String,
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
        val sqrtw = math.sqrt(w)
        val wx = gs * sqrtw
        val sj = resBc.value dot wx
        SkatTuple(sj * sj, wx, qtBc.value * wx)
      }

      val skatRDD = keyedRdd
        .map { case (k, vs) =>
          val vArray = vs.toArray.map { case (gs, w) => preProcessGenotypes(gs, w) }
          val skatStat = if (vArray.length.toLong * n < Int.MaxValue) {
            resultOp(vArray, sigmaSq)
          } else {
            largeNComputeSKATperGene(vArray, sigmaSq)
          }
          Row(k, skatStat.q, skatStat.pval, skatStat.fault)
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

    computeSkat(keyedRdd, keysType, y, cov, keyName, if (!useLargeN) computeSKATperGene else largeNComputeSKATperGene)
  }

  def keyedRDDSkat(vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    getGenotypes: (Iterable[Genotype], Int) => Vector[Double]):
  (RDD[(Any, Iterable[(Vector[Double], Double)])], Type) = {
    var mutableVds = vds
    val n = mutableVds.nSamples

    weightExpr match {
      case None => mutableVds = mutableVds.annotateVariantsExpr("va.AF = gs.callStats(g=> v).AF")
        .annotateVariantsExpr("va.weight = let af = if (va.AF[0] <= va.AF[1]) va.AF[0] else va.AF[1] in dbeta(af,1.0,25.0)**2")
      case _ =>
    }

    val (keysType, keysQuerier) = mutableVds.queryVA(variantKeys)
    val (weightType, weightQuerier) = weightExpr match {
      case None => mutableVds.queryVA("va.weight")
      case Some(expr) => mutableVds.queryVA(expr)
    }

    val typedWeightQuerier = weightType match {
      case _: TNumeric => (x: Annotation) => DoubleNumericConversion.to(weightQuerier(x))
      case _ => fatal("Weight must evaluate to numeric type")
    }

    val (keyType, keyIterator): (Type, Annotation => Iterator[Annotation]) = if (singleKey) {
      (keysType, (key: Any) => Iterator.single(key))
    } else {
      val keyType = keysType match {
        case t: TIterable => t.elementType
        case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
      }
      (keyType, (keys: Any) => keys.asInstanceOf[Iterable[Any]].iterator)
    }

    (mutableVds.rdd.flatMap { case (v, (va, gs)) =>
      (Option(keysQuerier(va)), Option(typedWeightQuerier(va))) match {
        case (Some(key), Some(w)) =>
          val gVector = getGenotypes(gs, n)
          keyIterator(key).map((_, gVector -> w))
        case _ => Iterator.empty
      }
    }.groupByKey(), keysType)
  }

  def computeSKATperGene(st: Array[SkatTuple], sigmaSq: Double): SkatStat = {
    val m = st.length
    val n = st(0).xw.size
    val k = st(0).qtxw.size

    var xwArray = new Array[Double](m * n)
    var qtxwArray = new Array[Double](m * k)
    var j = 0
    var i = 0

    //copy in non-zeros to weighted genotype matrix array
    i = 0
    while (i < m) {
      j = 0
      val xwsi = st(i).xw
      xwsi match {
        case dv: DenseVector[Double] =>
          while (j < n) {
            xwArray(i * n + j) = dv(j)
            j += 1
          }
          i += 1
        case sv: SparseVector[Double] =>
          val nnz = sv.used
          while (j < nnz) {
            val index = sv.index(j)
            xwArray(i * n + index) = sv.data(j)
            j += 1
          }
          i += 1
        case _ => fatal("SKAT is implemented for sparse or dense breeze vector datatypes.")
      }
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
    val zGramian = weightedGenotypes.t * weightedGenotypes
    val qtzGramian = qtWeightedGenotypes.t * qtWeightedGenotypes

    val SPG = new SkatModel(skatStat / (2 * sigmaSq))
    SPG.computeLinearSkatStats(zGramian, qtzGramian)
  }

  def largeNComputeSKATperGene(st: Array[SkatTuple], sigmaSq: Double): SkatStat = {
    val m = st.length
    val n = st(0).xw.size
    val k = st(0).qtxw.size

    val zGramianArray = new Array[Double](m * m)
    val qtzGramianArray = new Array[Double](m * m)

    var i = 0
    var j = 0

    while (i < m) {
      zGramianArray(i * m + i) = st(i).xw dot st(i).xw
      j = 0
      while (j < i) {
        val ijdotprod = st(i).xw dot st(j).xw
        zGramianArray(i * m + j) = ijdotprod
        zGramianArray(j * m + i) = ijdotprod
        j += 1
      }
      i += 1
    }

    i = 0
    while (i < m) {
      qtzGramianArray(i * m + i) = st(i).qtxw dot st(i).qtxw
      j = 0
      while (j < i) {
        val ijdotprod = st(i).qtxw dot st(j).qtxw
        qtzGramianArray(i * m + j) = ijdotprod
        qtzGramianArray(j * m + i) = ijdotprod
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

    val zGramian = new DenseMatrix[Double](m, m, zGramianArray)
    val qtzGramian = new DenseMatrix[Double](m, m, qtzGramianArray)

    val SPG = new SkatModel(skatStat / (2 * sigmaSq))
    SPG.computeLinearSkatStats(zGramian, qtzGramian)
  }
}
