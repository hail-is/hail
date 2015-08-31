package org.broadinstitute.k3.methods

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class PedigreeSuite extends TestNGSuite {
  @Test def testPedigree() {

    val file = "src/test/resources/sample_mendel.fam"

    val ped = Pedigree.read(file)

    assert(ped.trios.nIndiv == ped.nTrio)

    assert(ped.nFam == 3 && ped.nIndiv == 8 && ped.nSat(ped.isMale) == 5 && ped.nSat(ped.isFemale) == 3)

    assert(ped.nTrio == 3 && ped.nSat(ped.isTrio, ped.isMale) == 2 && ped.nSat(ped.isTrio, ped.isFemale) == 1 &&
      ped.nSat(ped.isTrio, ped.isCase) == 2 && ped.nSat(ped.isTrio, ped.isControl) == 1)

    assert(ped.nSat(ped.isTrio, ped.isCase, ped.isMale) == 1 && ped.nSat(ped.isTrio, ped.isCase, ped.isFemale) == 1 &&
      ped.nSat(ped.isTrio, ped.isControl, ped.isMale) == 1 && ped.nSat(ped.isTrio, ped.isControl, ped.isFemale) == 0)
  }
}
