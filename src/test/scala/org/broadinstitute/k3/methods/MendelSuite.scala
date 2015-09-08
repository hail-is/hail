package org.broadinstitute.k3.methods

import org.broadinstitute.k3.SparkSuite
import org.broadinstitute.k3.variant.Variant
import org.testng.annotations.Test

class MendelSuite extends SparkSuite {
  @Test def test() {
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam")
    val vds = LoadVCF(sc, "sparky", "src/test/resources/sample_mendel.vcf")
    val mds = MendelDataSet(vds, ped)

    val nPerFam = mds.nErrorPerFamily.collectAsMap()
    val nPerInd = mds.nErrorPerIndiv.collectAsMap()
    val nPerVar = mds.nErrorPerVariant.collectAsMap()

    val variant1 = Variant("1", 1, "C", "T")
    val variant2 = Variant("1", 2, "C", "T")
    val variant3 = Variant("X", 17, "C", "T")

    val s1 = vds.sampleIds.indexOf("Son1")
    val s2 = vds.sampleIds.indexOf("Daughter1")
    val s3 = vds.sampleIds.indexOf("Dad1")
    val s4 = vds.sampleIds.indexOf("Mom1")

    assert(nPerFam("Fam1") == 29)

    assert(nPerInd(s1) == 18)
    assert(nPerInd(s2) == 11)
    assert(nPerInd(s3) == 12)
    assert(nPerInd(s4) == 15)

    assert(nPerVar(variant1) == 2)
    assert(nPerVar(variant2) == 1)
    assert(nPerVar(variant3) == 2)

    mds.write("src/test/resources/sample_mendel.mendel")
    mds.writeVariant("src/test/resources/sample_mendel.lmendel")
    mds.writeFamily("src/test/resources/sample_mendel.fmendel")
    mds.writeIndiv("src/test/resources/sample_mendel.imendel")
  }
}
