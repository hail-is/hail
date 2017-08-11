package is.hail.stats

import breeze.linalg.{DenseVector, sum, _}
import breeze.stats.distributions._
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.{TArray, TFloat64, TInt32, TString, TStruct, TVariant}
import is.hail.utils._
import is.hail.variant.{GenomeReference, Genotype, VSMLocalValue, VSMMetadata, Variant, VariantDataset}
import org.apache.commons.math3.random.JDKRandomGenerator

object BaldingNicholsModel {

  def apply(hc: HailContext, nPops: Int, nSamples: Int, nVariants: Int,
    popDistArrayOpt: Option[Array[Double]], FstOfPopArrayOpt: Option[Array[Double]],
    seed: Int, nPartitionsOpt: Option[Int], af_dist: Distribution,
    gr: GenomeReference = GenomeReference.GRCh37): VariantDataset = {

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
        if (u.minVal < 0)
          fatal(s"minVal ${ u.minVal } must be at least 0")
        else if (u.maxVal > 1)
          fatal(s"maxVal ${ u.maxVal } must be at most 1")
      case _ =>
    }

    val N = nSamples
    val M = nVariants
    val K = nPops
    val popDist = DenseVector(popDistArray)
    val FstOfPop = DenseVector(FstOfPopArray)

    info(s"baldingnichols: generating genotypes for $K populations, $N samples, and $M variants...")

    Rand.generator.setSeed(seed)

    val popDist_k = popDist
    popDist_k :/= sum(popDist_k)

    val popDistRV = Multinomial(popDist_k)
    val popOfSample_n: DenseVector[Int] = DenseVector.fill[Int](N)(popDistRV.draw())
    val popOfSample_nBc = sc.broadcast(popOfSample_n)

    val Fst_k = FstOfPop
    val Fst1_k = (1d - Fst_k) :/ Fst_k
    val Fst1_kBc = sc.broadcast(Fst1_k)

    val rdd = sc.parallelize((0 until M).map(m => (m, Rand.randInt.draw())), nPartitions)
      .map { case (m, perVariantSeed) =>
        val perVariantRandomGenerator = new JDKRandomGenerator
        perVariantRandomGenerator.setSeed(perVariantSeed)
        val perVariantRandomBasis = new RandBasis(perVariantRandomGenerator)

        val ancestralAF = af_dist.getBreezeDist(perVariantRandomBasis).draw()

        val popAF_k = (0 until K).map { k =>
          new Beta(ancestralAF * Fst1_kBc.value(k), (1 - ancestralAF) * Fst1_kBc.value(k))(perVariantRandomBasis).draw()
        }

        (Variant("1", m + 1, "A", "C"),
          (Annotation(ancestralAF, popAF_k),
            (0 until N).map { n =>
              val p = popAF_k(popOfSample_nBc.value(n))
              val pSq = p * p
              val x = new Uniform(0, 1)(perVariantRandomBasis).draw()
              val gt =
                if (x < pSq)
                  2
                else if (x > 2 * p - pSq)
                  0
                else
                  1
              Genotype(gt)
            }: Iterable[Genotype]
          )
        )
      }
      .toOrderedRDD

    val sampleIds = (0 until N).map(_.toString).toArray
    val sampleAnnotations = (popOfSample_n.toArray: IndexedSeq[Int]).map(pop => Annotation(pop))

    val ancestralAFAnnotation = af_dist match {
      case UniformDist(minVal, maxVal) => Annotation("UniformDist", minVal, maxVal)
      case BetaDist(a, b) => Annotation("BetaDist", a, b)
      case TruncatedBetaDist(a, b, minVal, maxVal) => Annotation("TruncatedBetaDist", a, b, minVal, maxVal)
    }
    val globalAnnotation =
      Annotation(K, N, M, popDist_k.toArray: IndexedSeq[Double], Fst_k.toArray: IndexedSeq[Double], ancestralAFAnnotation, seed)

    val saSignature = TStruct("pop" -> TInt32)
    val vaSignature = TStruct("ancestralAF" -> TFloat64, "AF" -> TArray(TFloat64))

    val ancestralAFAnnotationSignature = af_dist match {
      case UniformDist(minVal, maxVal) => TStruct("type" -> TString, "minVal" -> TFloat64, "maxVal" -> TFloat64)
      case BetaDist(a, b) => TStruct("type" -> TString, "a" -> TFloat64, "b" -> TFloat64)
      case TruncatedBetaDist(a, b, minVal, maxVal) => TStruct("type" -> TString, "a" -> TFloat64, "b" -> TFloat64, "minVal" -> TFloat64, "maxVal" -> TFloat64)
    }

    val globalSignature = TStruct(
      "nPops" -> TInt32,
      "nSamples" -> TInt32,
      "nVariants" -> TInt32,
      "popDist" -> TArray(TFloat64),
      "Fst" -> TArray(TFloat64),
      "ancestralAFDist" -> ancestralAFAnnotationSignature,
      "seed" -> TInt32)
    new VariantDataset(hc,
      VSMMetadata(TString, saSignature, TVariant(gr), vaSignature, globalSignature, wasSplit = true),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd)
  }
}
