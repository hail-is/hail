package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, SafeIndexedSeq, SafeRow, ScalaToRegionValue, UnsafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PBaseStructTest extends HailSuite {
  @Test def testStructCopy() {
    def runTests(forceDeep: Boolean) {
      // add test for all primitives in face of force deep false
      // add test for downcast
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PArray(PInt32(true))), PStruct("a" -> PArray(PInt32())), Annotation(IndexedSeq(1,2,3)),
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(true)), PStruct("a" -> PInt64()), Annotation(12L),
        forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PArray(PInt32()), "b" -> PInt64(true)), PStruct("a" -> PArray(PInt32()), "b" -> PInt64()), Annotation(IndexedSeq(1,2,3), 3L),
        forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}
