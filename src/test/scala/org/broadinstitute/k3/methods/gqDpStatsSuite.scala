package org.broadinstitute.k3.methods

import math.abs
import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.k3.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.k3.utils.TestRDDBuilder
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.driver.VariantQC
import org.broadinstitute.k3.driver.SampleQC

class gqDpStatsSuite extends SparkSuite {
  @Test def test() {

    // Data here for pasting into R / Python
    // [[15,16,17,18,19,20,21,18], [25,25,26,28,15,30,15,20], [35,36,37,45,32,44,20,33], [20,21,13,23,25,27,16,22]]
    val arr = Array(Array(15,16,17,18,19,20,21,18), Array(25,25,26,28,15,30,15,20),
      Array(35,36,37,45,32,44,20,33), Array(20,21,13,23,25,27,16,22))

    val sampleMeans = Array(23.75, 24.5, 23.25, 28.5, 22.75, 30.25, 18.0, 23.25)
    val sampleDevs = Array(7.39509973, 7.36545993, 9.22970747, 10.16120072,
      6.41774883, 8.72854513, 2.54950976, 5.80409338)
    val variantMeans = Array(18.0 ,  23.0 , 35.25, 20.875)
    val variantDevs = Array(1.87082869, 5.33853913, 7.27581611, 4.28478413)

    // DP test first
    val dpVds = TestRDDBuilder.buildRDD(8, 4, sc, "sparky", dpArray=Some(arr), gqArray=None)
    val dpVariantR = VariantQC.results(dpVds, Array[AggregateMethod](dpStatCounterPer),
      Array(dpMeanPer, dpStDevPer))
    val dpSampleR = SampleQC.results(dpVds, Array[AggregateMethod](dpStatCounterPer),
      Array(dpMeanPer, dpStDevPer))
    dpVariantR.collect().foreach {
      case (v, a) =>
        println("Mean: %.2f/%.2f, Dev: %.2f/%.2f".format(a(1).asInstanceOf[Double], variantMeans(v.start), a(2)
          .asInstanceOf[Double], variantDevs(v.start)))
        assert(closeEnough(a(1).asInstanceOf[Double], variantMeans(v.start)))
        assert(closeEnough(a(2).asInstanceOf[Double], variantDevs(v.start)))
    }
    dpSampleR.foreach {
      case (s, a) =>
//          .asInstanceOf[Double], sampleDevs(s)))
        assert(closeEnough(a(1).asInstanceOf[Double], sampleMeans(s)))
        assert(closeEnough(a(2).asInstanceOf[Double], sampleDevs(s)))
    }

    // now test GQ
    val gqVds = TestRDDBuilder.buildRDD(8, 4, sc, "sparky", dpArray=Some(arr), gqArray=None)
    val gqVariantR = VariantQC.results(dpVds, Array[AggregateMethod](dpStatCounterPer),
      Array(dpMeanPer, dpStDevPer))
    val gqSampleR = SampleQC.results(dpVds, Array[AggregateMethod](dpStatCounterPer),
      Array(dpMeanPer, dpStDevPer))
    println(dpVariantR)
    dpVariantR.collect().foreach {
      case (v, a) =>
        assert(closeEnough(a(1).asInstanceOf[Double], variantMeans(v.start)))
        assert(closeEnough(a(2).asInstanceOf[Double], variantDevs(v.start)))
    }
    dpSampleR.foreach {
      case (s, a) =>
        assert(closeEnough(a(1).asInstanceOf[Double], sampleMeans(s)))
        assert(closeEnough(a(2).asInstanceOf[Double], sampleDevs(s)))
    }

  //    // check sample-wise
  //    for (i <- 0 until 8) {
  //      closeEnough(dpStatsBySample(i)._1, sampleMeans(i))
  //      closeEnough(dpStatsBySample(i)._2, sampleDevs(i))
  //    }
  //
  //    // check variant-wise
  //    dpStatsByVariant.collect().foreach { case (variant, (mean, dev)) =>
  //      assert(closeEnough(variantMeans(variant.start), mean))
  //      assert(closeEnough(variantDevs(variant.start), dev))
  //    }
  //
  //    // test GQ, let's use the same array for GQs instead of DP
  //    val vdsGQ = TestRDDBuilder.buildRDD(8, 4, sc, "sparky", dpArray=None, gqArray=Some(arr))
  //    val gqStatsBySample = GQStatsPerSample(vdsGQ)
  //    val gqStatsByVariant = GQStatsPerVariant(vdsGQ)
  //
  //    // check sample-wise
  //    for (i <- 0 until 8) {
  //      closeEnough(gqStatsBySample(i)._1, sampleMeans(i))
  //      closeEnough(gqStatsBySample(i)._2, sampleDevs(i))
  //    }
  //
  //    // check variant-wise
  //    gqStatsByVariant.collect().foreach { case (variant, (mean, dev)) =>
  //      assert(closeEnough(variantMeans(variant.start), mean))
  //      assert(closeEnough(variantDevs(variant.start), dev))
  //    }
}
}
