package org.broadinstitute.hail.io

import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Gen._
import sys.process._
import scala.language.postfixOps

class LoadBgenSuite extends SparkSuite {

/*  @Test def testUKBioBank10Variants() {
    val bgen = "/Users/jigold/ukbiobank_test10.bgen"
    val sampleFile = "/Users/jigold/ukbiobank_test10.sample"

    var s = State(sc, sqlContext, null)

    s = ImportBGEN.run(s, Array(bgen,"-s",sampleFile,"-n","5"))
    s = SplitMulti.run(s,Array[String]())
    s = ExportPlink.run(s, Array("-o","/tmp/testUkBiobank10var_hail"))
  }*/

  @Test def testMultipleUKBioBank10Variants() {
    val bgen = Array("/Users/jigold/ukbiobank_test10.chr1.bgen",
      "/Users/jigold/ukbiobank_test10.chr2.bgen",
      "/Users/jigold/ukbiobank_test10.chr3.bgen",
      "/Users/jigold/ukbiobank_test10.chr4.bgen",
      "/Users/jigold/ukbiobank_test10.chr5.bgen")
    val sampleFile = "/Users/jigold/ukbiobank_test10.sample"

    var s = State(sc, sqlContext, null)

    s = ImportBGEN.run(s,Array("-s",sampleFile,"-n","5","/Users/jigold/ukbiobank_test10.chr1.bgen","/Users/jigold/ukbiobank_test10.chr2.bgen"))
    s = SplitMulti.run(s,Array[String]())
    s = ExportPlink.run(s, Array("-o","/tmp/testUkBiobank10var_hail"))
  }
}