package is.hail.io

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.expr.types._
import is.hail.variant.{MatrixTable, VSMSubgen, _}
import org.testng.annotations.Test

class HardCallsSuite extends SparkSuite {
  @Test def test() {
    val p = forAll(MatrixTable.gen(hc, VSMSubgen.random)) { vds =>
      val hard = vds.selectEntries("g.GT")
      assert(hard.queryGenotypes("gs.map(g => g.GT).counter()") ==
        vds.queryGenotypes("gs.map(g => g.GT).counter()"))

      assert(hard.entryType == TStruct(
        "GT" -> TCall()))

      true
    }
    p.check()
  }
}
