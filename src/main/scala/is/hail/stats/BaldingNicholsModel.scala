package is.hail.stats

import breeze.linalg.{DenseVector, sum, _}
import breeze.stats.distributions._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types._
import is.hail.expr.MatrixLocalValue
import is.hail.rvd.OrderedRVD
import is.hail.utils._
import is.hail.variant.{GenomeReference, MatrixTable}
import org.apache.commons.math3.random.JDKRandomGenerator

object BaldingNicholsModel {

  def apply(hc: HailContext, nPops: Int, nSamples: Int, nVariants: Int,
    popDistArrayOpt: Option[Array[Double]], FstOfPopArrayOpt: Option[Array[Double]],
    seed: Int, nPartitionsOpt: Option[Int], af_dist: Distribution,
    gr: GenomeReference = GenomeReference.defaultReference): MatrixTable = {

    val sc = hc.sc

    if (nPops < 1)
      fatal(s"Number of populations must be positive, got ${ nPops }")

    if (nSamples < 1)
      fatal(s"Number of samples must be positive, got ${ nSamples }")

    if (nVariants < 1)
      fatal(s"Number of variants must be positive, got ${ nVariants }")

    val popDistArray = popDistArrayOpt.getOrElse(Array.fill[Double](nPops)(1d))

    if (popDistArray.length != nPops)
      fatal(s"Got ${ nPops } populations but ${ popDistArray.length } population ${ plural(popDistArray.length, "probability", "probabilities") }")
    popDistArray.foreach(p =>
      if (p < 0d)
        fatal(s"Population probabilities must be non-negative, got $p"))

    val FstOfPopArray = FstOfPopArrayOpt.getOrElse(Array.fill(nPops)(0.1))

    if (FstOfPopArray.length != nPops)
      fatal(s"Got ${ nPops } populations but ${ FstOfPopArray.length } ${ plural(FstOfPopArray.length, "value") }")

    FstOfPopArray.foreach(f =>
      if (f <= 0d || f >= 1d)
        fatal(s"F_st values must satisfy 0.0 < F_st < 1.0, got $f"))

    val nPartitions = nPartitionsOpt.getOrElse(Math.max((nSamples.toLong * nVariants / 1000000).toInt, 8))
    if (nPartitions < 1)
      fatal(s"Number of partitions must be >= 1, got $nPartitions")

    af_dist match {
      case u: UniformDist =>
        if (u.min < 0)
          fatal(s"min ${ u.min } must be at least 0")
        else if (u.max > 1)
          fatal(s"max ${ u.max } must be at most 1")
      case _ =>
    }

    val N = nSamples
    val M = nVariants
    val K = nPops
    val popDist = DenseVector(popDistArray)
    val FstOfPop = DenseVector(FstOfPopArray)

    info(s"balding_nichols_model: generating genotypes for $K populations, $N samples, and $M variants...")

    Rand.generator.setSeed(seed)

    val popDist_k = popDist
    popDist_k :/= sum(popDist_k)

    val popDistRV = Multinomial(popDist_k)
    val popOfSample_n: DenseVector[Int] = DenseVector.fill[Int](N)(popDistRV.draw())
    val popOfSample_nBc = sc.broadcast(popOfSample_n)

    val Fst_k = FstOfPop
    val Fst1_k = (1d - Fst_k) :/ Fst_k
    val Fst1_kBc = sc.broadcast(Fst1_k)

    val saSignature = TStruct("s" -> TString(), "pop" -> TInt32())
    val vaSignature = TStruct("ancestralAF" -> TFloat64(), "AF" -> TArray(TFloat64()))

    val ancestralAFAnnotation = af_dist match {
      case UniformDist(min, max) => Annotation("UniformDist", min, max)
      case BetaDist(a, b) => Annotation("BetaDist", a, b)
      case TruncatedBetaDist(a, b, min, max) => Annotation("TruncatedBetaDist", a, b, min, max)
    }
    val globalAnnotation =
      Annotation(K, N, M, popDistArray: IndexedSeq[Double], FstOfPopArray: IndexedSeq[Double], ancestralAFAnnotation, seed)

    val ancestralAFAnnotationSignature = af_dist match {
      case UniformDist(min, max) => TStruct("type" -> TString(), "min" -> TFloat64(), "max" -> TFloat64())
      case BetaDist(a, b) => TStruct("type" -> TString(), "a" -> TFloat64(), "b" -> TFloat64())
      case TruncatedBetaDist(a, b, min, max) => TStruct("type" -> TString(), "a" -> TFloat64(), "b" -> TFloat64(), "min" -> TFloat64(), "max" -> TFloat64())
    }

    val globalSignature = TStruct(
      "num_populations" -> TInt32(),
      "num_samples" -> TInt32(),
      "num_variants" -> TInt32(),
      "pop_dist" -> TArray(TFloat64()),
      "fst" -> TArray(TFloat64()),
      "ancestral_af_dist" -> ancestralAFAnnotationSignature,
      "seed" -> TInt32())

    val matrixType = MatrixType(
      globalType = globalSignature,
      colType = saSignature,
      colKey = Array("s"),
      vType = TVariant(gr),
      vaType = vaSignature,
      genotypeType = TStruct("GT" -> TCall()))

    val rowType = matrixType.rvRowType

    val rdd = sc.parallelize((0 until M).view.map(m => (m, Rand.randInt.draw())), nPartitions)
      .mapPartitions { it =>

        val region = Region()
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)

        it.map { case (m, perVariantSeed) =>
          val perVariantRandomGenerator = new JDKRandomGenerator
          perVariantRandomGenerator.setSeed(perVariantSeed)
          val perVariantRandomBasis = new RandBasis(perVariantRandomGenerator)

          val ancestralAF = af_dist.getBreezeDist(perVariantRandomBasis).draw()

          val popAF_k: IndexedSeq[Double] = Array.tabulate(K) { k =>
            new Beta(ancestralAF * Fst1_kBc.value(k), (1 - ancestralAF) * Fst1_kBc.value(k))(perVariantRandomBasis).draw()
          }

          region.clear()
          rvb.start(rowType)
          rvb.startStruct()

          // locus
          rvb.startStruct()
          rvb.addString("1")
          rvb.addInt(m + 1)
          rvb.endStruct()

          // variant
          rvb.startStruct()
          rvb.addString("1")
          rvb.addInt(m + 1)
          rvb.addString("A")
          rvb.startArray(1)
          rvb.startStruct()
          rvb.addString("A")
          rvb.addString("C")
          rvb.endStruct()
          rvb.endArray()
          rvb.endStruct()

          // va
          rvb.startStruct()
          rvb.addDouble(ancestralAF)
          rvb.startArray(popAF_k.length)
          var i = 0
          while (i < popAF_k.length) {
            rvb.addDouble(popAF_k(i))
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()

          // gs
          rvb.startArray(N)
          i = 0
          val unif = new Uniform(0, 1)(perVariantRandomBasis)
          while (i < N) {
            val p = popAF_k(popOfSample_nBc.value(i))
            val pSq = p * p
            val x = unif.draw()
            val gt =
              if (x < pSq)
                2
              else if (x > 2 * p - pSq)
                0
              else
                1

            rvb.startStruct()
            rvb.addInt(gt)
            rvb.endStruct()
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }

    val sampleAnnotations = (0 until N).map { i => Annotation(i.toString, popOfSample_n(i)) }.toArray

    // FIXME: should use fast keys
    val ordrdd = OrderedRVD(matrixType.orderedRVType, rdd, None, None)

    new MatrixTable(hc,
      matrixType,
      MatrixLocalValue(globalAnnotation, sampleAnnotations),
      ordrdd)
  }
}
