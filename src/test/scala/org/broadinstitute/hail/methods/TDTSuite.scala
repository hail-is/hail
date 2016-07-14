package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test


class TDTSuite extends SparkSuite {

  @Test def test() {
    // command option l : formats the line
    // command B -- go to declaration
    // CMD E -- open up recent files
    // CMD O -- browse classes (similar)
    // ctrl shift P -- show type
    // CMD R -- run

    //val vds2 = LoadVCF(sc, "src/test/resources/tdt_test3_6.vcf")
    //val vds2 = LoadVCF(sc, "/Users/jackkosmicki/Documents/SchizophreniaASD/fake_vcfTest3_6_fixed.vcf")
    val vds2 = LoadVCF(sc, "/Users/jackkosmicki/Documents/SchizophreniaASD/fake_vcf_forTDT_testing.vcf")
    val ped2 = Pedigree.read("src/test/resources/tdt_ped_test3_6.fam", sc.hadoopConfiguration, vds2.sampleIds)
    println(ped2.completeTrios.length)
    println(ped2.completeTrios(0))
    val tdt = TDT(vds2, ped2.completeTrios)
    tdt.foreach(println)
  }

}
