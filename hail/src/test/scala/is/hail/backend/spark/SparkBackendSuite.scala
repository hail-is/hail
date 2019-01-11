package is.hail.backend.spark

import is.hail.SparkSuite
import is.hail.expr.ir
import org.testng.annotations.Test

class SparkBackendSuite extends SparkSuite {

  @Test def testRangeCount() {
    val node = ir.ApplyBinaryPrimOp(ir.Add(), ir.TableCount(ir.TableRange(10, 2)), ir.TableCount(ir.TableRange(15, 5)))
    assert(SparkBackend.execute(hc.sc, node, optimize = false) == 25)
  }

}
