package org.broadinstitute.k3.methods

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class PedigreeSuite extends TestNGSuite {
  @Test def testPedigree() {

    val file = "src/test/resources/sample_mendel.fam"
    val file2 = "src/test/resources/sample_mendel2.fam"
    val ped = Pedigree.read(file)
    ped.write(file2)
    val ped2 = Pedigree.read(file2)
    assert(ped == ped2)

    assert(ped.trios.nIndiv == ped.nTrio)

    assert(ped.nFam == 4 && ped.nIndiv == 11)

    assert(ped.nSat(_.isMale) == 5 && ped.nSat(_.isFemale) == 5)

    assert(ped.nSat(_.isCase) == 4 && ped.nSat(_.isControl) == 3)

    assert(ped.nTrio == 3 && ped.nSat(_.isTrio, _.isMale) == 2 && ped.nSat(_.isTrio, _.isFemale) == 1 &&
      ped.nSat(_.isTrio, _.isCase) == 2 && ped.nSat(_.isTrio, _.isControl) == 1)

    assert(ped.nSat(_.isTrio, _.isCase, _.isMale) == 1 && ped.nSat(_.isTrio, _.isCase, _.isFemale) == 1 &&
      ped.nSat(_.isTrio, _.isControl, _.isMale) == 1 && ped.nSat(_.isTrio, _.isControl, _.isFemale) == 0)

  }
}
