package is.hail.stats

import breeze.stats._
import is.hail.SparkSuite
import is.hail.variant.{Call, Locus, Variant}
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import scala.collection.mutable

class BaldingNicholsModelSuite extends SparkSuite {

  @Test def testDeterminism() {
    val K = 3
    val N = 100
    val M = 100
    val popDist = Array(1d, 2d, 3d)
    val FstOfPop = Array(0.1, 0.2, 0.3)
    val seed = 0

    val bnm1 = BaldingNicholsModel(hc, K, N, M, Some(popDist), Some(FstOfPop), seed, Some(3), UniformDist(.1, .9))
    val bnm2 = BaldingNicholsModel(hc, K, N, M, Some(popDist), Some(FstOfPop), seed, Some(2), UniformDist(.1, .9))

    assert(bnm1.rdd.collect().toSeq == bnm2.rdd.collect().toSeq)
    assert(bnm1.globalAnnotation == bnm2.globalAnnotation)
    assert(bnm1.sampleAnnotations == bnm2.sampleAnnotations)

    bnm1.typecheck()
    bnm2.typecheck()
  }

  @Test def testDimensions()  {
    val K = 5
    val N = 10
    val M = 100
    val popDist = Array(1.0, 2.0, 3.0, 4.0, 5.0)
    val FstOfPop = Array(0.1, 0.2, 0.3, 0.2, 0.2)
    val seed = 0

    val bnm = BaldingNicholsModel(hc, K, N, M, Some(popDist), Some(FstOfPop), seed, Some(2), UniformDist(.1, .9))

    val gs_by_variant = bnm.rdd.collect.map(x => x._2._2)
    assert(gs_by_variant.forall(_.size == 10))

    assert(bnm.rdd.count() == 100)
  }

  @Test def testStats() {
    testStatsHelp(10, 100, 100, None, None, 0)
    testStatsHelp(40, 400, 20, None, None, 12)

    def testStatsHelp(K: Int, N: Int, M: Int, popDistOpt: Option[Array[Double]], FstOfPopOpt: Option[Array[Double]], seed: Int) = {
      val bnm = BaldingNicholsModel(hc, K, N, M, popDistOpt, FstOfPopOpt, seed, Some(4), UniformDist(.1, .9))

      val popDist: Array[Double] = popDistOpt.getOrElse(Array.fill(K)(1.0))

      //Test population distribution
      val popArray = bnm.sampleAnnotations.toArray.map(_.asInstanceOf[Row](0).toString.toDouble)
      val popCounts = popArray.groupBy(x => x).values.toSeq.sortBy(_(0)).map(_.size)

      popCounts.indices.foreach(index => {
        assertEquals(popCounts(index) / N, popDist(index), Math.ceil(N / K * .1))
      })

      //Test AF distributions
      val arrayOfVARows = bnm.variantsAndAnnotations.collect().map(_._2).toSeq.asInstanceOf[mutable.WrappedArray[Row]]
      val arrayOfVATuples = arrayOfVARows.map(row => (row.get(0), row.get(1)).asInstanceOf[(Double, Vector[Double])])

      val AFStats = arrayOfVATuples.map(tuple => meanAndVariance(tuple._2))

      arrayOfVATuples.map(_._1).zip(AFStats).foreach{
        case (p, mv) =>
          assertEquals(p, mv.mean, .2) //Consider alternatives to .2
          assertEquals(.1 * p * (1 - p), mv.variance, .1)
      }

      //Test genotype distributions
      val meanGeno_mk = bnm.typedRDD[Locus, Variant]
        .map(_._2._2.zip(popArray).groupBy(_._2).toSeq.sortBy(_._1))
        .map(_.map(popIterPair => mean(popIterPair._2.map(x => x._1.asInstanceOf[Row].getAs[Call](0).toDouble))).toArray)
        .collect()

      val p_mk = arrayOfVATuples.map(_._2.toArray)

      val meanDiff = (meanGeno_mk.flatten.sum - 2 * p_mk.flatten.sum) / (M * K)
      assertEquals(meanDiff, 0, .1)
    }
  }

  @Test def testAFRanges() {
    testRangeHelp(TruncatedBetaDist(.01, 2, .2, .8), .2, .8, 0)
    testRangeHelp(TruncatedBetaDist(3, 3, .4, .6), .4, .6, 1)

    testRangeHelp(UniformDist(.4, .7), .4, .7, 2)

    testRangeHelp(BetaDist(4, 6), 0, 1, 3)

    def testRangeHelp(dist: Distribution, min: Double, max: Double, seed: Int) {
      val bnm = BaldingNicholsModel(hc, 3, 400, 400, None, None, seed, Some(4), dist)

      val arrayOfVARows = bnm.variantsAndAnnotations.collect().map(_._2.asInstanceOf[Row]).toSeq
      val arrayOfAncestralAFs = arrayOfVARows.map(row => row.get(0).asInstanceOf[Double])

      assert(arrayOfAncestralAFs.forall(af => af > min && af < max))
    }
  }

}
