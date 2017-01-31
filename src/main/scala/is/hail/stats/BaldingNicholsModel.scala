package is.hail.stats

import breeze.linalg._
import breeze.stats.distributions._
import breeze.linalg.{DenseVector, sum}
import org.apache.spark.SparkContext
import org.apache.commons.math3.random.JDKRandomGenerator
import is.hail.annotations.Annotation
import is.hail.expr.{TArray, TDouble, TInt, TStruct}
import is.hail.utils._
import is.hail.variant.{Genotype, Variant, VariantDataset, VariantMetadata}

object BaldingNicholsModel {

  def apply(sc: SparkContext, nPops: Int, nSamples: Int, nVariants: Int,
    popDistArrayOpt: Option[Array[Double]], FstOfPopArrayOpt: Option[Array[Double]],
    seed: Int, nPartitionsOpt: Option[Int], af_dist: Distribution, root: String): VariantDataset = {

    if (nPops < 1)
      fatal(s"Number of populations must be positive, got ${ nPops }")

    if (nSamples < 1)
      fatal(s"Number of samples must be positive, got ${ nSamples }")

    if (nVariants < 1)
      fatal(s"Number of variants must be positive, got ${ nVariants }")

    val popDistArray = popDistArrayOpt.getOrElse(Array.fill[Double](nPops)(1d))

    if (popDistArray.size != nPops)
      fatal(s"Got ${ nPops } populations but ${ popDistArray.size } population ${ plural(popDistArray.size, "probability", "probabilities") }")
    popDistArray.foreach(p =>
      if (p < 0d)
        fatal(s"Population probabilities must be non-negative, got $p"))

    val FstOfPopArray = FstOfPopArrayOpt.getOrElse(Array.fill(nPops)(0.1))

    if (FstOfPopArray.size != nPops)
      fatal(s"Got ${ nPops } populations but ${ FstOfPopArray.size } ${ plural(FstOfPopArray.size, "value") }")

    FstOfPopArray.foreach(f =>
      if (f <= 0d || f >= 1d)
        fatal(s"F_st values must satisfy 0.0 < F_st < 1.0, got $f"))

    val nPartitions = nPartitionsOpt.getOrElse(Math.max(nSamples * nVariants / 1000000, 8))
    if (nPartitions <= 1)
      fatal(s"Number of partitions must be positive, got $nPartitions")

    if (af_dist.isInstanceOf[UniformDist]) {
      val uniform = af_dist.asInstanceOf[UniformDist]
      if (uniform.minVal < 0)
        fatal("minVal cannot be less than 0")
      else if (uniform.maxVal > 1)
        fatal("maxVal cannot be greater than 1")
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

    val variantSeed = Rand.randInt.draw();
    val variantSeedBc = sc.broadcast(variantSeed)
    

    val rdd = sc.parallelize(
      (0 until M), nPartitions).map { m =>
        val perVariantSeed = variantSeedBc.value + m
        val perVariantRandomGenerator = new JDKRandomGenerator
        perVariantRandomGenerator.setSeed(perVariantSeed)
        val perVariantRandomBasis = new RandBasis(perVariantRandomGenerator)

        val ancestralAF = af_dist.getBreezeDist(perVariantRandomBasis).draw()

        val popAF_k = (0 until K).map{k =>
          new Beta(ancestralAF * Fst1_kBc.value(k), (1 - ancestralAF) * Fst1_kBc.value(k))(perVariantRandomBasis).draw()
        }

        (Variant("1", m + 1, "A", "C"),
          (Annotation(Annotation(ancestralAF, popAF_k)),
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
    val sampleAnnotations = (popOfSample_n.toArray: IndexedSeq[Int]).map(pop => Annotation(Annotation(pop)))
    val globalAnnotation = Annotation(
      Annotation(K, N, M, popDist_k.toArray: IndexedSeq[Double], Fst_k.toArray: IndexedSeq[Double], seed))

    val saSignature = TStruct(root -> TStruct("pop" -> TInt))
    val vaSignature = TStruct(root -> TStruct("ancestralAF" -> TDouble, "AF" -> TArray(TDouble)))
    val globalSignature = TStruct(root ->
      TStruct("nPops" -> TInt, "nSamples" -> TInt, "nVariants" -> TInt,
        "popDist" -> TArray(TDouble), "Fst" -> TArray(TDouble), "seed" -> TInt))

    new VariantDataset(
      new VariantMetadata(sampleIds, sampleAnnotations, globalAnnotation, saSignature, vaSignature, globalSignature, wasSplit=true),
      rdd
    )
  }
}
