package org.broadinstitute.k3.methods

import org.broadinstitute.k3.SparkSuite
import org.testng.annotations.Test

class PedigreeSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "sparky", "src/test/resources/sample_mendel.vcf")
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam", vds.sampleIds)
    ped.write("/tmp/sample_mendel.fam", vds.sampleIds)  // FIXME: this is not right
    val pedwr = Pedigree.read("/tmp/sample_mendel.fam", vds.sampleIds)
    assert(ped == pedwr)

    val pedBothParents = new Pedigree(ped.trioMap.filter{ case (k,t) => t.hasDadMom })
    assert(pedBothParents.nIndiv == ped.nCompleteTrio)

    assert(ped.nFam == 5 && ped.nIndiv == 11)
    assert(ped.nSatisfying(_.isMale) == 5 && ped.nSatisfying(_.isFemale) == 5)
    assert(ped.nSatisfying(_.isCase) == 4 && ped.nSatisfying(_.isControl) == 3)
    assert(ped.nCompleteTrio == 3 &&
      ped.nSatisfying(_.hasDadMom, _.isMale) == 2 && ped.nSatisfying(_.hasDadMom, _.isFemale) == 1 &&
      ped.nSatisfying(_.hasDadMom, _.isCase) == 2 && ped.nSatisfying(_.hasDadMom, _.isControl) == 1)
    assert(ped.nSatisfying(_.hasDadMom, _.isCase, _.isMale) == 1 &&
      ped.nSatisfying(_.hasDadMom, _.isCase, _.isFemale) == 1 &&
      ped.nSatisfying(_.hasDadMom, _.isControl, _.isMale) == 1 &&
      ped.nSatisfying(_.hasDadMom, _.isControl, _.isFemale) == 0)

  }
}
