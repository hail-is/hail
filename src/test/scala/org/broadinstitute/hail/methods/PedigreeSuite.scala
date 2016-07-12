package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.check.Prop._
import org.testng.annotations.Test


class PedigreeSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/pedigree.vcf")
    val ped = Pedigree.read("src/test/resources/pedigree.fam", sc.hadoopConfiguration, vds.sampleIds)
    val f = tmpDir.createTempFile("pedigree", ".fam")
    ped.write(f, sc.hadoopConfiguration)
    val pedwr = Pedigree.read(f, sc.hadoopConfiguration, vds.sampleIds)
    assert(ped.trios.sameElements(pedwr.trios)) // this passes because all samples in .fam are in pedigree.vcf

    val nuclearFams = Pedigree.nuclearFams(ped.completeTrios)
    val sampleIndex = vds.sampleIds.zipWithIndex.toMap
    assert(nuclearFams(("Dad1", "Mom1")).toSet ==
      Set("Son1", "Dtr1"))
    assert(nuclearFams(("Dad2", "Mom2")).toSet ==
      Set("Son2"))
    assert(nuclearFams.size == 2 && ped.completeTrios.length == 3 && ped.trios.length == 11)

    assert(ped.nSatisfying(_.isMale) == 6 && ped.nSatisfying(_.isFemale) == 5)

    assert(ped.nSatisfying(_.isCase) == 4 && ped.nSatisfying(_.isControl) == 3)

    assert(ped.nSatisfying(_.isComplete, _.isMale) == 2 && ped.nSatisfying(_.isComplete, _.isFemale) == 1 &&
      ped.nSatisfying(_.isComplete, _.isCase) == 2 && ped.nSatisfying(_.isComplete, _.isControl) == 1)

    assert(ped.nSatisfying(_.isComplete, _.isCase, _.isMale) == 1 &&
      ped.nSatisfying(_.isComplete, _.isCase, _.isFemale) == 1 &&
      ped.nSatisfying(_.isComplete, _.isControl, _.isMale) == 1 &&
      ped.nSatisfying(_.isComplete, _.isControl, _.isFemale) == 0)

    val ped2 = Pedigree.read("src/test/resources/pedigreeWithExtraSample.fam", sc.hadoopConfiguration, vds.sampleIds)

    assert(ped.trios.toSet == ped2.trios.toSet)

    // FIXME: How to test
    // ped.writeSummary("/tmp/pedigree.sumfam", sc.hadoopConfiguration)
  }

  @Test def generated() {

    val p = forAll(Pedigree.genWithIds()) { case (ids: IndexedSeq[String], ped: Pedigree) =>
      val f = tmpDir.createTempFile(extension = ".fam")
      ped.write(f, hadoopConf)
      val ped2 = Pedigree.read(f, hadoopConf, ids)
      (ped.trios: IndexedSeq[Trio]) == (ped2.trios: IndexedSeq[Trio])
    }

    p.check()
  }
}
