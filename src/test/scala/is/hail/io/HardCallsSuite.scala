package is.hail.io

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.expr.{TCall, TStruct}
import is.hail.variant.{VSMSubgen, VariantSampleMatrix, _}
import org.testng.annotations.Test

class HardCallsSuite extends SparkSuite {
  @Test def test() {
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random)) { vds =>
      val hard = vds.hardCalls()
      assert(hard.queryGenotypes("gs.map(g => g.GT).counter()") ==
        vds.queryGenotypes("gs.map(g => g.GT).counter()"))

      assert(hard.genotypeSignature == TStruct(
        "GT" -> TCall()))

      true
    }
    p.check()
  }
}
