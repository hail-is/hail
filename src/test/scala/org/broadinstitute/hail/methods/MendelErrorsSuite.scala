package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

class MendelErrorsSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample_mendel.vcf")
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam", vds.sampleIds)
    val men = MendelErrors(vds, ped)

    men.mendelErrors.collect().foreach(println)

    val nPerFam = men.nErrorPerNuclearFamily.collectAsMap()
    val nPerIndiv = men.nErrorPerIndiv.collectAsMap()
    val nPerVariant = men.nErrorPerVariant.collectAsMap()

    val son = vds.sampleIds.indexOf("Son1")
    val dtr = vds.sampleIds.indexOf("Daughter1")
    val dad = vds.sampleIds.indexOf("Dad1")
    val mom = vds.sampleIds.indexOf("Mom1")
    val dad2 = vds.sampleIds.indexOf("Dad2")
    val mom2 = vds.sampleIds.indexOf("Mom2")

    val variant1 = Variant("1", 1, "C", "T")
    val variant2 = Variant("1", 2, "C", "T")
    val variant3 = Variant("X", 1, "C", "T")
    val variant4 = Variant("X", 3, "C", "T")
    val variant5 = Variant("20", 1, "C", "T")

    assert(nPerFam.size == 2)
    assert(nPerIndiv.size == 11)
    assert(nPerVariant.size == 22)

    assert(nPerFam((dad, mom)) == 34)
    assert(nPerFam((dad2, mom2)) == 0)

    assert(nPerIndiv(son) == 20)
    assert(nPerIndiv(dtr) == 14)
    assert(nPerIndiv(dad) == 15)
    assert(nPerIndiv(mom) == 18)
    assert(nPerIndiv(dad2) == 0)

    assert(nPerVariant(variant1) == 2)
    assert(nPerVariant(variant2) == 1)
    assert(nPerVariant(variant3) == 2)
    assert(nPerVariant(variant4) == 1)
    assert(nPerVariant.get(variant5).isEmpty)

    //FIXME: How to test these?
    //men.writeMendel("/tmp/sample_mendel.mendel")
    //men.writeMendelL("/tmp/sample_mendel.lmendel")
    //men.writeMendelF("/tmp/sample_mendel.fmendel")
    //men.writeMendelI("/tmp/sample_mendel.imendel")
  }
}
