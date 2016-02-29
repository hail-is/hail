package org.broadinstitute.hail.variant


import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.methods.{LoadVCF, LinearRegressionFromHardCallSet, CovariateData, Pedigree}

import org.testng.annotations.Test

class HardCallSetProfileSuite extends SparkSuite {
  @Test def test() {

    /*
    val hcs = HardCallSet.read(sqlContext, "/Users/jbloom/t2d/GoT2D.chr22.15.hcs")
    val ped = Pedigree.read("/Users/jbloom/t2d/GoT2D.fam", sc.hadoopConfiguration, hcs.sampleIds)
    val cov = CovariateData.read("/Users/jbloom/t2d/GoT2D.cov", sc.hadoopConfiguration, hcs.sampleIds)
      .filterSamples(ped.phenotypedSamples)

    hcs.cache()

    println("Run 1 starting...")
    val linreg = LinearRegressionFromHardCallSet(hcs, ped, cov)
    println("Run 1 done.")

    println("Run 2 starting...")
    val linreg2 = LinearRegressionFromHardCallSet(hcs, ped, cov)
    println("Run 2 done.")
    */



    // val vds = LoadVCF(sc, "/Users/jbloom/t2d/GoT2D.chr22.final_integrated_snps_indels_sv_beagle_thunder_2874_20121121.vcf.bgz")
    // var state = State(sc, sqlContext, vds)
    // state = SplitMulti.run(state, Array[String]())
    // state = WriteHardCallSet.run(state,
    //  Array("-o", "/Users/jbloom/t2d/GoT2D.chr22.15.idea.hcs", "-f", "/Users/jbloom/t2d/GoT2D.fam", "-c", "/Users/jbloom/t2d/GoT2D.cov"))

    var state = State(sc, sqlContext)

    state = ReadHcs.run(state, Array("-i", "/Users/jbloom/t2d/GoT2D.chr22.15.idea.hcs"))

    state = CacheHcs.run(state, Array[String]())
    state = LinearRegressionFromHardCallSetCommand.run(state,
      Array("-f", "/Users/jbloom/t2d/GoT2D.fam", "-c", "/Users/jbloom/t2d/GoT2D.cov", "-o", "/tmp/GoT2D.chr22.linreg"))
    state = LinearRegressionFromHardCallSetCommand.run(state,
      Array("-f", "/Users/jbloom/t2d/GoT2D.fam", "-c", "/Users/jbloom/t2d/GoT2D.cov", "-o", "/tmp/GoT2D.chr22.linreg"))

    println("All done!")
  }

}
