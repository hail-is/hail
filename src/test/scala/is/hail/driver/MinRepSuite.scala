package is.hail.driver

import is.hail.SparkSuite
import is.hail.variant.Variant
import org.testng.annotations.Test

class MinRepSuite extends SparkSuite{

  @Test def minrep() {

    assert(Variant("1",10, "TAA", Array("TA")).minrep.compare(Variant("1",10, "TA", Array("T"))) == 0)
    assert(Variant("1",10, "ACTG", Array("ACT")).minrep.compare(Variant("1",12, "TG", Array("T"))) == 0)
    assert(Variant("1",10, "AAACAAAC", Array("AAAC")).minrep.compare(Variant("1",10, "AAACA", Array("A"))) == 0)
    assert(Variant("1",10, "AATAA", Array("AAGAA")).minrep.compare(Variant("1",12, "T", Array("G"))) == 0)

    assert(Variant("1",10, "TAA", Array("TA","TTA")).minrep
      .compare(Variant("1",10, "TA", Array("T","TT"))) == 0)
    assert(Variant("1",10, "GCTAA", Array("GCAAA","G")).minrep
      .compare(Variant("1",10, "GCTAA", Array("GCAAA","G"))) == 0)
    assert(Variant("1",10, "GCTAA", Array("GCAAA","GCCAA")).minrep
      .compare(Variant("1",12, "T", Array("A","C"))) == 0)

  }


}
