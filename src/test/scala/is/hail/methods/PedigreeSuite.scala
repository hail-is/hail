package is.hail.methods

import is.hail.SparkSuite
import is.hail.check.Prop._
import org.testng.annotations.Test


class PedigreeSuite extends SparkSuite {
  @Test def test() {
    val vds = hc.importVCF("src/test/resources/pedigree.vcf")
    val ped = Pedigree.read("src/test/resources/pedigree.fam", sc.hadoopConfiguration).filterTo(vds.sampleIds.toSet)
    val f = tmpDir.createTempFile("pedigree", ".fam")
    ped.write(f, sc.hadoopConfiguration)
    val pedwr = Pedigree.read(f, sc.hadoopConfiguration).filterTo(vds.sampleIds.toSet)
    assert(ped.trios == pedwr.trios) // this passes because all samples in .fam are in pedigree.vcf

    val nuclearFams = Pedigree.nuclearFams(ped.completeTrios)
    val sampleIndex = vds.sampleIds.zipWithIndex.toMap
    assert(nuclearFams(("Dad1", "Mom1")).toSet ==
      Set("Son1", "Dtr1"))
    assert(nuclearFams(("Dad2", "Mom2")).toSet ==
      Set("Son2"))
    assert(nuclearFams.size == 2 && ped.completeTrios.length == 3 && ped.trios.length == 11)

    assert(ped.nSatisfying(_.isMale) == 6 && ped.nSatisfying(_.isFemale) == 5)

    val ped2 = Pedigree.read("src/test/resources/pedigreeWithExtraSample.fam", sc.hadoopConfiguration)
      .filterTo(vds.sampleIds.toSet)

    assert(ped.trios.toSet == ped2.trios.toSet)
  }

  @Test def generated() {

    val p = forAll(Pedigree.genWithIds()) { case (ids: IndexedSeq[String], ped: Pedigree) =>
      val f = tmpDir.createTempFile("pedigree", ".fam")
      ped.write(f, hadoopConf)
      val ped2 = Pedigree.read(f, hadoopConf)
      (ped.trios: IndexedSeq[Trio]) == (ped2.trios: IndexedSeq[Trio])
    }

    p.check()
  }
}
