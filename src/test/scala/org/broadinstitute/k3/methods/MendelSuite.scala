package org.broadinstitute.k3.methods

import org.broadinstitute.k3.SparkSuite
import org.broadinstitute.k3.variant.Variant
import org.testng.annotations.Test

class MendelSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "sparky", "src/test/resources/sample_mendel.vcf")
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam", vds.sampleIds)
    val men = MendelErrors(vds, ped)

    val nPerFam = men.nErrorPerNuclearFamily.collectAsMap()
    val nPerInd = men.nErrorPerIndiv.collectAsMap()
    val nPerVar = men.nErrorPerVariant.collectAsMap()

    val son = vds.sampleIds.indexOf("Son1")
    val dtr = vds.sampleIds.indexOf("Daughter1")
    val dad = vds.sampleIds.indexOf("Dad1")
    val mom = vds.sampleIds.indexOf("Mom1")

    val variant1 = Variant("1", 1, "C", "T")
    val variant2 = Variant("1", 2, "C", "T")
    val variant3 = Variant("X", 17, "C", "T")

    //assert(nPerFam("Fam1") == 29)

    assert(nPerInd(son) == 18)
    assert(nPerInd(dtr) == 11)
    assert(nPerInd(dad) == 12)
    assert(nPerInd(mom) == 15)

    assert(nPerVar(variant1) == 2)
    assert(nPerVar(variant2) == 1)
    assert(nPerVar(variant3) == 2)

    men.writeMendel("src/test/resources/sample_mendel.mendel")
    men.writeMendelL("src/test/resources/sample_mendel.lmendel")
    men.writeMendelF("src/test/resources/sample_mendel.fmendel")
    men.writeMendelI("src/test/resources/sample_mendel.imendel")
  }
}
