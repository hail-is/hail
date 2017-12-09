package is.hail.vds

import is.hail.SparkSuite
import is.hail.variant.{GenomeReference, Variant}
import org.testng.annotations.Test

class MinRepSuite extends SparkSuite {

  @Test def minrep() {
    val gr = GenomeReference.GRCh37

    assert(Variant("1", 10, "TAA", Array("TA")).minRep.compare(Variant("1", 10, "TA", Array("T")), gr) == 0)
    assert(Variant("1", 10, "ACTG", Array("ACT")).minRep.compare(Variant("1", 12, "TG", Array("T")), gr) == 0)
    assert(Variant("1", 10, "AAACAAAC", Array("AAAC")).minRep.compare(Variant("1", 10, "AAACA", Array("A")), gr) == 0)
    assert(Variant("1", 10, "AATAA", Array("AAGAA")).minRep.compare(Variant("1", 12, "T", Array("G")), gr) == 0)
    assert(Variant("1", 10, "AATAA", Array("*")).minRep.compare(Variant("1", 10, "A", Array("*")), gr) == 0)

    assert(Variant("1", 10, "TAA", Array("TA", "TTA")).minRep
      .compare(Variant("1", 10, "TA", Array("T", "TT")), gr) == 0)
    assert(Variant("1", 10, "GCTAA", Array("GCAAA", "G")).minRep
      .compare(Variant("1", 10, "GCTAA", Array("GCAAA", "G")), gr) == 0)
    assert(Variant("1", 10, "GCTAA", Array("GCAAA", "GCCAA")).minRep
      .compare(Variant("1", 12, "T", Array("A", "C")), gr) == 0)
    assert(Variant("1", 10, "GCTAA", Array("GCAAA", "GCCAA", "*")).minRep
      .compare(Variant("1", 12, "T", Array("A", "C", "*")), gr) == 0)
  }
}
