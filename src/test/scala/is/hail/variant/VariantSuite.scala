package is.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class VariantSuite extends TestNGSuite {
  @Test def test() {
    assert(Contig.compare("3", "18") < 0)
    assert(Contig.compare("18", "3") > 0)
    assert(Contig.compare("7", "7") == 0)

    assert(Contig.compare("3", "X") < 0)
    assert(Contig.compare("X", "3") > 0)
    assert(Contig.compare("X", "X") == 0)

    assert(Contig.compare("X", "Y") < 0)
    assert(Contig.compare("Y", "X") > 0)
    assert(Contig.compare("Y", "MT") < 0)

    assert(Contig.compare("18", "SPQR") < 0)
    assert(Contig.compare("MT", "SPQR") < 0)

  }
}
