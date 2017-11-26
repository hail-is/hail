package is.hail.stats

import breeze.linalg.{DenseVector, sum, _}
import breeze.stats.distributions._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{MatrixType, TArray, TCall, TFloat64, TInt32, TLocus, TString, TStruct, TVariant}
import is.hail.rvd.OrderedRVD
import is.hail.utils._
import is.hail.variant.{GenomeReference, Genotype, VSMLocalValue, VSMMetadata, Variant, VariantDataset, VariantSampleMatrix}
import org.apache.commons.math3.random.JDKRandomGenerator

object BaldingNicholsModel {

  def apply(hc: HailContext, nPops: Int, nSamples: Int, nVariants: Int,
    popDistArrayOpt: Option[Array[Double]], FstOfPopArrayOpt: Option[Array[Double]],
    seed: Int, nPartitionsOpt: Option[Int], af_dist: Distribution,
    gr: GenomeReference = GenomeReference.defaultReference): VariantDataset = {

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

    info(s"balding nichols: generating genotypes for $K populations, $N samples, and $M variants...")

    Rand.generator.setSeed(seed)

    val popDist_k = popDist
    popDist_k :/= sum(popDist_k)

    val popDistRV = Multinomial(popDist_k)
    val popOfSample_n: DenseVector[Int] = DenseVector.fill[Int](N)(popDistRV.draw())
    val popOfSample_nBc = sc.broadcast(popOfSample_n)

    val Fst_k = FstOfPop
    val Fst1_k = (1d - Fst_k) :/ Fst_k
    val Fst1_kBc = sc.broadcast(Fst1_k)

    val rowType = TStruct(
      "pk" -> TLocus(gr),
      "v" -> TVariant(gr),
      "va" -> TStruct("ancestralAF" -> TFloat64(), "AF" -> TArray(TFloat64())),
      "gs" -> !TArray(TStruct("GT" -> TCall())))


    val rdd = sc.parallelize((0 until M).map(m => (m, Rand.randInt.draw())), nPartitions)
      .mapPartitions { it =>

        val rvb = new RegionValueBuilder(MemoryBuffer())

        it.map { case (m, perVariantSeed) =>
          val perVariantRandomGenerator = new JDKRandomGenerator
          perVariantRandomGenerator.setSeed(perVariantSeed)
          val perVariantRandomBasis = new RandBasis(perVariantRandomGenerator)

          val ancestralAF = af_dist.getBreezeDist(perVariantRandomBasis).draw()

          val popAF_k = (0 until K).map { k =>
            new Beta(ancestralAF * Fst1_kBc.value(k), (1 - ancestralAF) * Fst1_kBc.value(k))(perVariantRandomBasis).draw()
          }

          rvb.clear()
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
          while (i < N) {
            val p = popAF_k(popOfSample_nBc.value(i))
            val pSq = p * p
            val x = new Uniform(0, 1)(perVariantRandomBasis).draw()
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
          rvb.result()
        }
      }

    // TODO make orderedrdd2

    val sampleIds = (0 until N).map(_.toString).toArray
    val sampleAnnotations = (popOfSample_n.toArray: IndexedSeq[Int]).map(pop => Annotation(pop))

    val ancestralAFAnnotation = af_dist match {
      case UniformDist(minVal, maxVal) => Annotation("UniformDist", minVal, maxVal)
      case BetaDist(a, b) => Annotation("BetaDist", a, b)
      case TruncatedBetaDist(a, b, minVal, maxVal) => Annotation("TruncatedBetaDist", a, b, minVal, maxVal)
    }
    val globalAnnotation =
      Annotation(K, N, M, popDist_k.toArray: IndexedSeq[Double], Fst_k.toArray: IndexedSeq[Double], ancestralAFAnnotation, seed)

    val saSignature = TStruct("pop" -> TInt32())
    val vaSignature = TStruct("ancestralAF" -> TFloat64(), "AF" -> TArray(TFloat64()))

    val ancestralAFAnnotationSignature = af_dist match {
      case UniformDist(minVal, maxVal) => TStruct("type" -> TString(), "minVal" -> TFloat64(), "maxVal" -> TFloat64())
      case BetaDist(a, b) => TStruct("type" -> TString(), "a" -> TFloat64(), "b" -> TFloat64())
      case TruncatedBetaDist(a, b, minVal, maxVal) => TStruct("type" -> TString(), "a" -> TFloat64(), "b" -> TFloat64(), "minVal" -> TFloat64(), "maxVal" -> TFloat64())
    }

    val globalSignature = TStruct(
      "nPops" -> TInt32(),
      "nSamples" -> TInt32(),
      "nVariants" -> TInt32(),
      "popDist" -> TArray(TFloat64()),
      "Fst" -> TArray(TFloat64()),
      "ancestralAFDist" -> ancestralAFAnnotationSignature,
      "seed" -> TInt32())

    val vsmMetadata = VSMMetadata(
      TString(),
      saSignature,
      TVariant(gr),
      vaSignature,
      TStruct.empty(),
      TStruct("GT" -> TCall()))

    val matrixType = MatrixType(vsmMetadata)

    // FIXME: should use fast keys
    val ordrdd = OrderedRVD(matrixType.orderedRVType, rdd, None, None)

    new VariantDataset(hc,
      vsmMetadata,
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations),
      ordrdd)
  }
}
