package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.VariantQC
import org.broadinstitute.hail.driver.SampleQC

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
    val dpSingletons = sSingletonVariants(dpVds)
    val dpVariantR = VariantQC.results(dpVds)
    val dpSampleR = SampleQC.results(dpVds, dpSingletons)
    
    dpVariantR.collect().foreach {
      case (v, a) =>
//        println("Mean: Computed=%.2f, True=%.2f | Dev: Computed=%.2f, True=%.2f".format(a(0).asInstanceOf[Option[Double]].get,
//          variantMeans(v.start),a(1).asInstanceOf[Option[Double]].get, variantDevs(v.start)))
        assert(closeEnough(a.dpSC.mean, variantMeans(v.start)))
        assert(closeEnough(a.dpSC.stdev, variantDevs(v.start)))
    }

      dpSampleR.collect().foreach {
      case (s, a) =>
//        println("Mean: Computed=%.2f, True=%.2f | Dev: Computed=%.2f, True=%.2f".format(a(1).asInstanceOf[Double], sampleMeans(s), a(2)
//          .asInstanceOf[Double], sampleDevs(s)))
        assert(closeEnough(a.dpSC.mean, sampleMeans(s)))
        assert(closeEnough(a.dpSC.stdev, sampleDevs(s)))
    }

    // now test GQ
    val gqVds = TestRDDBuilder.buildRDD(8, 4, sc, "sparky", dpArray=None, gqArray=Some(arr))
    val gqSingletons = sSingletonVariants(gqVds)
    val gqVariantR = VariantQC.results(gqVds)
    val gqSampleR = SampleQC.results(gqVds, gqSingletons)

    gqVariantR.collect().foreach {
      case (v, a) =>
//        println("Mean: Computed=%.2f, True=%.2f | Dev: Computed=%.2f, True=%.2f".format(a(1).asInstanceOf[Double], variantMeans(v.start), a(2)
//          .asInstanceOf[Double], variantDevs(v.start)))

        assert(closeEnough(a.gqSC.mean, variantMeans(v.start)))
        assert(closeEnough(a.gqSC.stdev, variantDevs(v.start)))
    }

    gqSampleR.collect().foreach {
      case (s, a) =>
//        println("Mean: Computed=%.2f, True=%.2f | Dev: Computed=%.2f, True=%.2f".format(a(1).asInstanceOf[Double], sampleMeans(s), a(2)
//          .asInstanceOf[Double], sampleDevs(s)))
        assert(closeEnough(a.gqSC.mean, sampleMeans(s)))
        assert(closeEnough(a.gqSC.stdev, sampleDevs(s)))
    }
  }
}
