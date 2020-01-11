package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PNDArraySuite extends HailSuite {
  @Test def copyyTests() {
    def runTests(forceDeep: Boolean) {
      PhysicalTestUtils.copyTestExecutor(PNDArray(PInt64(true), 1), PNDArray(PInt64(true), 1), Annotation(0, 1, Annotation(1L), Annotation(1L), IndexedSeq(4L,5L,6L)),
        forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}
