package is.hail.vds

import is.hail.SparkSuite
import is.hail.variant.Variant
import org.testng.annotations.Test

class MinRepSuite extends SparkSuite {

  @Test def minrep() {

    assert(Variant("1", 10, "TAA", Array("TA")).minRep.compare(Variant("1", 10, "TA", Array("T"))) == 0)
    assert(Variant("1", 10, "ACTG", Array("ACT")).minRep.compare(Variant("1", 12, "TG", Array("T"))) == 0)
    assert(Variant("1", 10, "AAACAAAC", Array("AAAC")).minRep.compare(Variant("1", 10, "AAACA", Array("A"))) == 0)
    assert(Variant("1", 10, "AATAA", Array("AAGAA")).minRep.compare(Variant("1", 12, "T", Array("G"))) == 0)
    assert(Variant("1", 10, "AATAA", Array("*")).minRep.compare(Variant("1", 10, "A", Array("*"))) == 0)

    assert(Variant("1", 10, "TAA", Array("TA", "TTA")).minRep
      .compare(Variant("1", 10, "TA", Array("T", "TT"))) == 0)
    assert(Variant("1", 10, "GCTAA", Array("GCAAA", "G")).minRep
      .compare(Variant("1", 10, "GCTAA", Array("GCAAA", "G"))) == 0)
    assert(Variant("1", 10, "GCTAA", Array("GCAAA", "GCCAA")).minRep
      .compare(Variant("1", 12, "T", Array("A", "C"))) == 0)
    assert(Variant("1", 10, "GCTAA", Array("GCAAA", "GCCAA", "*")).minRep
      .compare(Variant("1", 12, "T", Array("A", "C", "*"))) == 0)
  }
}
