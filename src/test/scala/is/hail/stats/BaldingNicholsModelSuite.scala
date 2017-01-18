package is.hail.stats

import is.hail.SparkSuite
import is.hail.variant.{Genotype, VariantDataset}
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.testng.annotations.Test
import org.testng.Assert.assertEquals
import breeze.stats._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

class BaldingNicholsModelSuite extends SparkSuite {


  @Test def testDeterminism() = {
    val K = 3
    val N = 2
    val M = 2
    val popDist = Array(1d, 2d, 3d)
    val FstOfPop = Array(.1, .2, .3)
    val seed = 0

    val bnm1 = BaldingNicholsModel(sc, K, N, M, popDist, FstOfPop, seed, 2, "bn")
    val bnm2 = BaldingNicholsModel(sc, K, N, M, popDist, FstOfPop, seed, 2, "bn")

    assert(bnm1.rdd.collect().toSeq == bnm2.rdd.collect().toSeq)
    assert(bnm1.globalAnnotation == bnm2.globalAnnotation)
  }

  @Test def testDimensions() = {
    val K = 5
    val N = 10
    val M = 100
    val popDist = Array(1.0, 2.0, 3.0, 4.0, 5.0)
    val FstOfPop = Array(0.1, 0.2, 0.3, 0.2, 0.2)
    val seed = 0

    val bnm: VariantDataset = BaldingNicholsModel(sc, K, N, M, popDist, FstOfPop, seed, 2, "bn")

    //Check right number of samples
    val allCorrectSize = bnm.rdd.collect.map(x => x._2._2).forall(x => x.size == 10)
    assert(allCorrectSize)

    //Check right number of variants
    assert(bnm.rdd.count() == 100)
  }

  @Test def testStats() {
    testStatsHelp(4, 4000, 10, Array(1.0, 1.0, 1.0, 1.0), Array(.1, .1, .1, .1), 90)
  }

  def testStatsHelp(K: Int, N: Int, M: Int, popDist: Array[Double], FstOfPop:Array[Double], seed: Int) = {
    val bnm = BaldingNicholsModel(sc, K, N, M, popDist, FstOfPop, seed, 4, "bn")

    //TEST population Distribution
    val populationArray = bnm.sampleAnnotations.toArray.map(_.toString.toInt)
    println(populationArray.mkString(","))
    val populationCounts = populationArray.groupBy(x => x).values.toSeq.sortBy(_(0)).map(_.size)

    populationCounts.indices.foreach(index => {
      assertEquals(populationCounts(index) / N, popDist(index), Math.ceil(N / K * .1))
    })


    //Testing subpopulation AF Distributions
    val arrayOfVARows = bnm.variantsAndAnnotations.collect().map(_._2).toSeq.asInstanceOf[mutable.WrappedArray[GenericRow]]
    val arrayOfVATuples = arrayOfVARows.map(row => row.get(0).asInstanceOf[GenericRow]).map(row => (row.get(0), row.get(1)).asInstanceOf[(Double, Vector[Double])])

    val AFStats = arrayOfVATuples.map(tuple => meanAndVariance(tuple._2))

    arrayOfVATuples.map(_._1).zip(AFStats).foreach({
      case (p, mv) => {
        assertEquals(p, mv.mean, .2) //Consider alternatives to .2
        assertEquals(.1 * p * (1 - p), mv.variance, .1)
      }
    })

    //Testing Genotype means
    val meanGeno_mk = bnm.rdd
      .map(_._2._2.zip(populationArray).groupBy(_._2).toSeq.sortBy(_._1))
      .map(_.map(popIterPair => mean(popIterPair._2.map(_._1.gt.get.toDouble))).toArray)
      .collect()

    val p_mk = arrayOfVATuples.map(_._2.toArray)

    val meanDiff = (meanGeno_mk.flatten.sum - 2 * p_mk.flatten.sum) / (M * K)
    assertEquals(meanDiff, 0, .1)

  }

  /*@Test def testPopulationDistribution() = {
    //Testing assuming default settings for user
    val K1 = 4
    val N1 = 400
    val M1 = 80
    val popDist1 = Array(1.0, 1.0, 1.0, 1.0)
    val FstOfPop1 = Array(.1, .1, .1, .1)
    val seed1 = 200

    val bnm1: VariantDataset = BaldingNicholsModel(sc, K1, N1, M1, popDist1, FstOfPop1, seed1, 4, "bn")

    val populationArray1: Array[Int] = bnm1.sampleAnnotations.toArray.map(_.toString.toInt)
    val populationCounts: Iterable[Int] = populationArray1.groupBy(x => x).values.map(_.size)

    populationCounts.foreach(x => assertEquals(x, 100, 10))


    //Testing with some more interesting settings
    val K2 = 3
    val N2 = 600
    val M2 = 90
    val popDist2 = Array(1.0, 2.0, 3.0)
    val FstOfPop2 = Array(.1, .3, .4)
    val seed2 = 23456

    val bnm2: VariantDataset = BaldingNicholsModel(sc, K2, N2, M2, popDist2, FstOfPop2, seed2, 4, "bn")

    val populationArray2: Array[Int] = bnm2.sampleAnnotations.toArray.map(_.toString.toInt)
    val populationCounts2 = populationArray2.groupBy(x => x).values.toSeq.sortWith((a, b) => a(0) < b(0)).map(_.size)

    populationCounts2.indices.foreach(index => {
      assertEquals(populationCounts2(index) / N2, popDist2(index), 20)
    })
  }*/

  /*@Test def testSubpopulationAlleleFrequencyDistributions() = {
    val K = 20
    val N = 800
    val M = 1000
    val popDist = Array.fill[Double](20)(1)
    val FstOfPop = Array.fill[Double](20)(.1)
    val seed = -432


    val bnm: VariantDataset = BaldingNicholsModel(sc, K, N, M, popDist, FstOfPop, seed, 4, "bn")

    val arrayOfRows = bnm.variantsAndAnnotations.collect().map(_._2).toSeq.asInstanceOf[mutable.WrappedArray[GenericRow]]
    val arrayOfTuples = arrayOfRows.map(row => row.get(0).asInstanceOf[GenericRow]).map(row => (row.get(0), row.get(1)).asInstanceOf[(Double, Vector[Double])])

    arrayOfTuples.foreach((t: (Double, Vector[Double])) => {
      val m = mean(t._2)
      val v = variance(t._2)

      assertEquals(m, t._1, .15)
      assertEquals(v, .1 * m * (1 - m), .05)
    })

  }*/

  @Test def testGenotypeMean() = {
    val K1 = 4
    val N1 = 400
    val M1 = 80
    val popDist1 = Array(1.0, 1.0, 1.0, 1.0)
    val FstOfPop1 = Array(.1, .1, .1, .1)
    val seed1 = 200

    val bnm1: VariantDataset = BaldingNicholsModel(sc, K1, N1, M1, popDist1, FstOfPop1, seed1, 4, "bn")

    val populationArray1: Array[Int] = bnm1.sampleAnnotations.toArray.map(_.toString.toInt)

    val genotypePopulationPairs: RDD[Map[Int, Iterable[(Genotype, Int)]]] = bnm1.rdd.map(_._2._2).map(genotypes => genotypes.zip(populationArray1).groupBy(_._2))

    genotypePopulationPairs.foreach(map => map.foreach(popGenoPair => {

    }))
  }

}
