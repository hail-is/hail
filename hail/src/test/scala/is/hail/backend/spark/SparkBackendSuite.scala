package is.hail.backend.spark

import is.hail.SparkSuite
import is.hail.expr.ir
import org.testng.annotations.Test

class SparkBackendSuite extends SparkSuite {

  @Test def testRangeCount() {
    val node = ir.TableCount(ir.TableRange(10, 2))
    assert(SparkBackend.execute(hc.sc, node, optimize = false) == 10)
  }

}
