package is.hail.expr.ir

import is.hail.expr.types.virtual.{TFloat64, TInt32, TTuple}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class FoldConstantsSuite extends TestNGSuite {
  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64())
    assert(FoldConstants(x) == x)
  }
}
