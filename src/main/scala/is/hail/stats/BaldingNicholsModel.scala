package is.hail.stats

import breeze.linalg.{DenseVector, sum, _}
import breeze.stats.distributions._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types._
import is.hail.rvd.{OrderedRVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant.{Call2, MatrixTable, ReferenceGenome}
import org.apache.commons.math3.random.JDKRandomGenerator
import org.apache.spark.sql.Row

object BaldingNicholsModel {

  def apply(hc: HailContext,
    nPops: Int,
    nSamples: Int,
    nVariants: Int,
    popDistArrayOpt: Option[Array[Double]],
    FstOfPopArrayOpt: Option[Array[Double]],
    seed: Int,
    nPartitionsOpt: Option[Int],
    af_dist: Distribution,
    rg: ReferenceGenome = ReferenceGenome.defaultReference,
    mixture: Boolean = false): MatrixTable = {

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
    val popDist = DenseVector(popDistArray.clone())
    val FstOfPop = DenseVector(FstOfPopArray)

    info(s"balding_nichols_model: generating genotypes for $K populations, $N samples, and $M variants...")

    Rand.generator.setSeed(seed)

    val popDist_k = popDist
    val popOfSample_n = DenseMatrix.zeros[Double](if (mixture) K else 1, N)
    
    if (mixture) {
      val popDistRV = Dirichlet(popDist_k)
      (0 until N).foreach(j => popOfSample_n(::, j) := popDistRV.draw())
    } else {
      popDist_k :/= sum(popDist_k)
      val popDistRV = Multinomial(popDist_k)
      (0 until N).foreach(j => popOfSample_n(0, j) = popDistRV.draw())
    }
    
    val popOfSample_nBc = sc.broadcast(popOfSample_n)

    val Fst_k = FstOfPop
    val Fst1_k = (1d - Fst_k) /:/ Fst_k
    val Fst1_kBc = sc.broadcast(Fst1_k)

    val saSignature = TStruct("sample_idx" -> TInt32(), "pop" -> (if (mixture) TArray(TFloat64()) else TInt32()))
    val vaSignature = TStruct("ancestralAF" -> TFloat64(), "AF" -> TArray(TFloat64()))

    val ancestralAFAnnotation = af_dist match {
      case UniformDist(min, max) => Annotation("UniformDist", min, max)
      case BetaDist(a, b) => Annotation("BetaDist", a, b)
      case TruncatedBetaDist(a, b, min, max) => Annotation("TruncatedBetaDist", a, b, min, max)
    }
    val globalAnnotation =
      Row(K, N, M, popDistArray: IndexedSeq[Double], FstOfPopArray: IndexedSeq[Double], ancestralAFAnnotation, seed, mixture)

    val ancestralAFAnnotationSignature = af_dist match {
      case UniformDist(min, max) => TStruct("type" -> TString(), "min" -> TFloat64(), "max" -> TFloat64())
      case BetaDist(a, b) => TStruct("type" -> TString(), "a" -> TFloat64(), "b" -> TFloat64())
      case TruncatedBetaDist(a, b, min, max) => TStruct("type" -> TString(), "a" -> TFloat64(), "b" -> TFloat64(), "min" -> TFloat64(), "max" -> TFloat64())
    }

    val globalSignature = TStruct(
      "n_populations" -> TInt32(),
      "n_samples" -> TInt32(),
      "n_variants" -> TInt32(),
      "pop_dist" -> TArray(TFloat64()),
      "fst" -> TArray(TFloat64()),
      "ancestral_af_dist" -> ancestralAFAnnotationSignature,
      "seed" -> TInt32(),
      "mixture" -> TBoolean())

    val matrixType: MatrixType = MatrixType.fromParts(
      globalType = globalSignature,
      colType = saSignature,
      colKey = Array("sample_idx"),
      rowType = TStruct("locus" -> TLocus(rg), "alleles" -> TArray(TString())) ++ vaSignature,
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = TStruct("GT" -> TCall()))

    val rvType = matrixType.rvRowType

    val rdd = ContextRDD.weaken[RVDContext](sc.parallelize((0 until M).view.map(m => (m, Rand.randInt.draw())), nPartitions))
      .cmapPartitions { (ctx, it) =>
        val region = ctx.region
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)

        it.map { case (m, perVariantSeed) =>
          val perVariantRandomGenerator = new JDKRandomGenerator
          perVariantRandomGenerator.setSeed(perVariantSeed)
          val perVariantRandomBasis = new RandBasis(perVariantRandomGenerator)

          val ancestralAF = af_dist.getBreezeDist(perVariantRandomBasis).draw()

          val popAF_k: DenseVector[Double] = DenseVector(
            Array.tabulate(K) { k =>
              new Beta(ancestralAF * Fst1_kBc.value(k), (1 - ancestralAF) * Fst1_kBc.value(k))(perVariantRandomBasis)
                .draw()
          })

          rvb.start(rvType)
          rvb.startStruct()

          // locus
          rvb.startStruct()
          rvb.addString("1")
          rvb.addInt(m + 1)
          rvb.endStruct()

          // alleles
          rvb.startArray(2)
          rvb.addString("A")
          rvb.addString("C")
          rvb.endArray()

          // va
          rvb.addDouble(ancestralAF)
          rvb.startArray(popAF_k.length)
          var i = 0
          while (i < popAF_k.length) {
            rvb.addDouble(popAF_k(i))
            i += 1
          }
          rvb.endArray()

          // gs
          rvb.startArray(N)
          i = 0
          val unif = new Uniform(0, 1)(perVariantRandomBasis)
          while (i < N) {
            val p =
              if (mixture)
                popOfSample_nBc.value(::, i) dot popAF_k
              else
                popAF_k(popOfSample_nBc.value(0, i).toInt)
            val pSq = p * p
            val x = unif.draw()
            val c =
              if (x < pSq)
                Call2.fromUnphasedDiploidGtIndex(2)
              else if (x > 2 * p - pSq)
                Call2.fromUnphasedDiploidGtIndex(0)
              else
                Call2.fromUnphasedDiploidGtIndex(1)

            rvb.startStruct()
            rvb.addInt(c)
            rvb.endStruct()
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }

    val sampleAnnotations: Array[Annotation] =
      if (mixture)
        Array.tabulate(N)(i => Annotation(i, popOfSample_n(::, i).data.toIndexedSeq))
      else
        Array.tabulate(N)(i => Annotation(i, popOfSample_n(0, i).toInt))

    // FIXME: should use fast keys
    val ordrdd = OrderedRVD.coerce(matrixType.orvdType, rdd, None, None)

    new MatrixTable(hc,
      matrixType,
      BroadcastRow(globalAnnotation, matrixType.globalType, hc.sc),
      BroadcastIndexedSeq(sampleAnnotations, TArray(matrixType.colType), hc.sc),
      ordrdd)
  }
}
