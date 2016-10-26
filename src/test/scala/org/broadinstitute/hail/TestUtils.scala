package org.broadinstitute.hail

import breeze.linalg._
import breeze.stats.distributions.{Beta, Binomial, Multinomial, Uniform, RandBasis}
import org.apache.commons.math3.random.JDKRandomGenerator
import org.broadinstitute.hail.utils._
import org.apache.spark.SparkContext
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.variant.{Genotype, Variant, VariantDataset, VariantMetadata}

object TestUtils {

  import org.scalatest.Assertions._

  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[FatalException](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    if (!p)
      println(
        s"""expected fatal exception with pattern `$regex'
           |  Found: ${thrown.getMessage}""".stripMargin)
    assert(p)
  }

  // G(i,j) is genotype of variant j in sample i encoded as -1, 0, 1, 2
  def vdsFromMatrix(sc: SparkContext)(G: Matrix[Int], nPartitions: Int = sc.defaultMinPartitions): VariantDataset = {
    val sampleIds = (0 until G.rows).map(_.toString).toArray

    val rdd = sc.parallelize(
      (0 until G.cols).map { j =>
        (Variant("1", j + 1, "A", "C"),
          (Annotation.empty,
            (0 until G.rows).map { i =>
              val gt = G(i, j)
              assert(gt >= -1 && gt <= 2)
              Genotype(gt)
            }: Iterable[Genotype]
          )
        )
      },
      nPartitions
    ).toOrderedRDD

    new VariantDataset(VariantMetadata(sampleIds), rdd)
  }

  // K populations, N samples, M variants, pi is K-vector proportional to population sizes, F is K-vector of F_st values
  def baldingNichols(K: Int, N: Int, M: Int, seed: Int = 0,
    piVect: Option[DenseVector[Double]] = None,
    FVect: Option[DenseVector[Double]] = None): DenseMatrix[Int] = {

    require(piVect.forall(_.length == K))
    require(piVect.forall(_.forall(_ >= 0d)))
    require(FVect.forall(_.length == K))
    require(FVect.forall(_.forall(Fk => Fk > 0d && Fk < 1d)))

    val gen = new JDKRandomGenerator()
    gen.setSeed(seed)

    implicit val rand: RandBasis = new RandBasis(gen)

    val pi = Multinomial(piVect.getOrElse(DenseVector.fill[Double](K)(1.0)))
    val k_n = DenseVector.fill[Int](N)(pi.draw())

    val F = FVect.getOrElse(DenseVector.fill[Double](K)(0.1))
    val F1 = (1d - F) :/ F

    val P0 = Uniform(0.1, 0.9)
    val p0_m = DenseVector.fill[Double](M)(P0.draw())

    val p_km = DenseMatrix.zeros[Double](K, M)

    var k = 0
    var m = 0
    while (k < K) {
      m = 0
      while (m < M) {
        p_km(k,m) = new Beta((1 - p0_m(m)) * F1(k), p0_m(m) * F1(k)).draw()
        m += 1
      }
      k += 1
    }

    val g_nm = DenseMatrix.zeros[Int](N,M)

    var n = 0
    while (n < N) {
      m = 0
      while (m < M) {
        val p_nm = p_km(k_n(n), m)
        g_nm(n, m) = Binomial(2, p_nm).draw()
        m += 1
      }
      n += 1
    }

    g_nm
  }
}

