package org.broadinstitute.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions._
import org.apache.commons.math3.random.JDKRandomGenerator
import org.broadinstitute.hail.utils.info
import org.broadinstitute.hail.variant.VariantDataset
import org.apache.spark.SparkContext
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{TArray, TDouble, TInt, TStruct}

object BaldingNicholsModel {
  // K populations, N samples, M variants
  // popDist is K-vector proportional to population distribution
  // FstOfPop is K-vector of F_st values
  def apply(K: Int, N: Int, M: Int,
    popDistOpt: Option[DenseVector[Double]] = None,
    FstOfPopOpt: Option[DenseVector[Double]] = None,
    seed: Option[Int] = None): BaldingNicholsModel = {

    require(K > 0)
    require(N > 0)
    require(M > 0)
    require(popDistOpt.forall(_.length == K))
    require(popDistOpt.forall(_.forall(_ >= 0d)))
    require(FstOfPopOpt.forall(_.length == K))
    require(FstOfPopOpt.forall(_.forall(f => f > 0d && f < 1d)))

    info(s"baldingnichols: generating genotypes for $K populations, $N samples, and $M variants...")

    val gen = new JDKRandomGenerator()
    seed.foreach(gen.setSeed)
    implicit val rand: RandBasis = new RandBasis(gen)

    val popDist_k = popDistOpt.getOrElse(DenseVector.fill[Double](K)(1d))
    popDist_k :/= sum(popDist_k)
    val popDistRV = Multinomial(popDist_k)
    val popOfSample_n = DenseVector.fill[Int](N)(popDistRV.draw())

    val Fst_k = FstOfPopOpt.getOrElse(DenseVector.fill[Double](K)(0.1))
    val Fst1_k = (1d - Fst_k) :/ Fst_k

    val ancestralAF = Uniform(0.1, 0.9)
    val ancestralAF_m = DenseVector.fill[Double](M)(ancestralAF.draw())

    val popAF_km = DenseMatrix.zeros[Double](K, M)
    (0 until K).foreach(k =>
      (0 until M).foreach(m =>
        popAF_km(k,m) = new Beta((1 - ancestralAF_m(m)) * Fst1_k(k), ancestralAF_m(m) * Fst1_k(k)).draw()))

    val unif = rand.uniform

    val genotype_nm = DenseMatrix.zeros[Int](N,M)
    (0 until N).foreach(n =>
      (0 until M).foreach { m =>
        val p = popAF_km(popOfSample_n(n), m)
        val pSq = p * p
        val x = unif.draw()
        genotype_nm(n, m) =
          if (x < pSq)
            0
          else if (x > 2 * p - pSq) // equiv to 1 - (1 - p)^2
            2
          else
            1
      }
    )

    BaldingNicholsModel(K, N, M, genotype_nm, popOfSample_n, ancestralAF_m, popAF_km, popDist_k, Fst_k, seed)
  }
}

case class BaldingNicholsModel(
  nPops: Int,
  nSamples: Int,
  nVariants: Int,
  genotypes: DenseMatrix[Int],
  popOfSample: DenseVector[Int],
  ancestralAF: DenseVector[Double],
  popAF: DenseMatrix[Double],
  popDist: DenseVector[Double],
  FstOfPop: DenseVector[Double],
  seed: Option[Int]) {

  require(genotypes.rows == nSamples)
  require(genotypes.cols == nVariants)
  require(popOfSample.size == nSamples)
  require(ancestralAF.size == nVariants)
  require(popAF.rows == nPops)
  require(popAF.cols == nVariants)
  require(popDist.size == nPops)
  require(FstOfPop.size == nPops)

  def toVDS(sc: SparkContext, root: String = "bn", nPartitions: Option[Int]): VariantDataset = {
    val globalHead = s"${Annotation.GLOBAL_HEAD}.$root"
    val sampleToPop = (0 until nSamples).map(i => (i.toString, popOfSample(i))).toMap

    val vds = vdsFromMatrix(sc)(genotypes, None, nPartitions.getOrElse(sc.defaultMinPartitions))

    val freqSchema = TStruct(("ancAF", TDouble) +: (0 until nPops).map(i => (s"AF$i", TDouble)): _*)

    val (newVAS, inserter) = vds.insertVA(freqSchema, root)

    val ancestralAFBc = sc.broadcast(ancestralAF)
    val popAFBc = sc.broadcast(popAF)

    seed.map(s => vds.annotateGlobal(s, TInt, s"$globalHead.seed")).getOrElse(vds)
      .annotateGlobal(nPops, TInt, s"$globalHead.nPops")
      .annotateGlobal(nSamples, TInt, s"$globalHead.nSamples")
      .annotateGlobal(nVariants, TInt, s"$globalHead.nVariants")
      .annotateGlobal(popDist.toArray: IndexedSeq[Double], TArray(TDouble), s"$globalHead.popDist")
      .annotateGlobal(FstOfPop.toArray: IndexedSeq[Double], TArray(TDouble), s"$globalHead.Fst")
      .annotateSamples(sampleToPop, TInt, s"${Annotation.SAMPLE_HEAD}.$root.pop")
      .mapAnnotations{ case (v, va, gs) =>
        val ancfreq = ancestralAFBc.value(v.start - 1)
        val freq =  popAFBc.value(::, v.start - 1).toArray
        inserter(va, Some(Annotation.fromSeq(ancfreq +: freq)))
    }.copy(vaSignature = newVAS)
  }
}

