package org.broadinstitute.hail.io

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class LoadBgenSuite extends SparkSuite {
  @Test def test() {
    val munge1a = BgenLoader.mungeIndel("13:104613359:A_AAT", "R", "I")
    val munge1b = BgenLoader.mungeIndel("13:104613359:A_AAT", "I", "R")
    println("From 13:104613359:A_AAT, R, I => %s, %s".format(munge1a._1, munge1a._2))
    println("From 13:104613359:A_AAT, I, R => %s, %s".format(munge1b._1, munge1b._2))
    val munge2a = BgenLoader.mungeIndel("13:104620197:TGAA_", "R", "D")
    val munge2b = BgenLoader.mungeIndel("13:104620197:TGAA_", "D", "R")
    println("From 13:104620197:TGAA_, R, D => %s, %s".format(munge2a._1, munge2a._2))
    println("From 13:104620197:TGAA_, D, R => %s, %s".format(munge2b._1, munge2b._2))
//    val loader = BgenLoader("src/test/resources/11_36760324_51539425.bgen", sc)
    val loader = BgenLoader("src/test/resources/biggerM.bgen", sc)
//    val loader = BgenLoader("src/test/resources/indelgen.bgen", sc)
//    val loader = BgenLoader("src/test/resources/bigM.bgen", sc)
//    loader.makeVDS(sc, "sparky")
  }
}