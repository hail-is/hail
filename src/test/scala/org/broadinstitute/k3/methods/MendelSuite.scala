package org.broadinstitute.k3.methods

import org.broadinstitute.k3.SparkSuite
import org.testng.annotations.Test

class MendelSuite extends SparkSuite {
  @Test def test() {
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam")
    val vds = LoadVCF(sc, "sparky", "src/test/resources/sample_mendel.vcf")
    val mds = MendelDataSet(vds, ped)

    mds.write("src/test/resources/sample_mendel.mendel")
    mds.writeVariant("src/test/resources/sample_mendel.lmendel")
    mds.writeFamily("src/test/resources/sample_mendel.fmendel")
    mds.writeIndiv("src/test/resources/sample_mendel.imendel")
  }
}
