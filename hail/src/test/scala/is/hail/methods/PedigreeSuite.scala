package is.hail.methods

import is.hail.{HailSuite, TestUtils}
import is.hail.check.Prop._
import org.testng.annotations.Test


class PedigreeSuite extends HailSuite {
  @Test def test() {
    val vds = TestUtils.importVCF(hc, "src/test/resources/pedigree.vcf")
    val ped = Pedigree.read("src/test/resources/pedigree.fam", sFS).filterTo(vds.stringSampleIdSet)
    val f = tmpDir.createTempFile("pedigree", ".fam")
    ped.write(f, sFS)
    val pedwr = Pedigree.read(f, sFS).filterTo(vds.stringSampleIdSet)
    assert(ped.trios == pedwr.trios) // this passes because all samples in .fam are in pedigree.vcf

    val nuclearFams = Pedigree.nuclearFams(ped.completeTrios)
    val sampleIndex = vds.stringSampleIds.zipWithIndex.toMap
    assert(nuclearFams(("Dad1", "Mom1")).toSet ==
      Set("Son1", "Dtr1"))
    assert(nuclearFams(("Dad2", "Mom2")).toSet ==
      Set("Son2"))
    assert(nuclearFams.size == 2 && ped.completeTrios.length == 3 && ped.trios.length == 11)

    assert(ped.nSatisfying(_.isMale) == 6 && ped.nSatisfying(_.isFemale) == 5)

    val ped2 = Pedigree.read("src/test/resources/pedigreeWithExtraSample.fam", sFS)
      .filterTo(vds.stringSampleIdSet)

    assert(ped.trios.toSet == ped2.trios.toSet)
  }

  @Test def generated() {

    val p = forAll(Pedigree.genWithIds()) { case (ids: IndexedSeq[String], ped: Pedigree) =>
      val f = tmpDir.createTempFile("pedigree", ".fam")
      ped.write(f, sFS)
      val ped2 = Pedigree.read(f, sFS)
      (ped.trios: IndexedSeq[Trio]) == (ped2.trios: IndexedSeq[Trio])
    }

    p.check()
  }
}
