package is.hail.expr.types.physical

import is.hail.HailSuite
import org.testng.annotations.Test

class PSubsetStructSuite extends HailSuite {
  @Test def testSubsetStruct(): Unit = {
    // 1) Define a PCanonicalStruct
    // 2) Use SRVB to build the struct off-heap
    // 3) Create a 2nd PStruct that is a subset of the first
    // 4) Use SRVB to build that struct
    // 5) Use PSubsetStruct
    // Compare values 3 and 5
  }
}
