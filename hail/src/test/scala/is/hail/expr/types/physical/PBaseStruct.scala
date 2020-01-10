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
      PhysicalTestUtils.copyTestExecutor(PStruct(), PStruct(), Annotation(),
        forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64()), PStruct("a" -> PInt64()), Annotation(12L),
        forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(true)), PStruct("a" -> PInt64(true)), Annotation(11L),
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(false)), PStruct("a" -> PInt64(true)), Annotation(3L),
        expectCompileErr = true, forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(false)), PStruct("a" -> PInt64(true)), Annotation(14L),
        allowDowncast = true, forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(false)), PStruct("a" -> PInt64(true)), Annotation(null),
        allowDowncast = true, expectRuntimeErr = true, forceDeep = forceDeep)

      var srcType = PStruct("a" -> PInt64(true), "b" -> PInt32(true), "c" -> PFloat64(true), "d" -> PFloat32(true), "e" -> PBoolean(true))
      var destType = PStruct("a" -> PInt64(true), "b" -> PInt32(true), "c" -> PFloat64(true), "d" -> PFloat32(true), "e" -> PBoolean(true))
      var expectedVal = Annotation(13L, 12, 13.0, 10.0F, true)

      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, forceDeep = forceDeep)

      srcType = PStruct("a" -> srcType, "c" -> PFloat32())
      destType = PStruct("a" -> destType, "c" -> PFloat32())
      var nestedExpectedVal = Annotation(expectedVal, 13.0F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, forceDeep = forceDeep)

      srcType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      destType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, forceDeep = forceDeep)

      srcType = PStruct("a" -> srcType, "b" -> PFloat32())
      destType = PStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 14.0F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, forceDeep = forceDeep)

      srcType = PStruct("a" -> PInt64(), "b" -> PInt32(true), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      destType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, forceDeep = forceDeep)

      srcType = PStruct("a" -> srcType, "b" -> PFloat32())
      destType = PStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 15.0F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, forceDeep = forceDeep)

      srcType = PStruct("a" -> PInt64(), "b" -> PInt32(true), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      destType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(true), "d" -> PFloat32(), "e" -> PBoolean())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, expectCompileErr = true, forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, allowDowncast = true, forceDeep = forceDeep)

      srcType = PStruct("a" -> srcType, "b" -> PFloat32())
      destType = PStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 13F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, expectCompileErr = true, forceDeep = forceDeep)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, allowDowncast = true, forceDeep = forceDeep)

      srcType = PStruct("a" -> PArray(PInt32(true)), "b" -> PInt64())
      destType = PStruct("a" -> PArray(PInt32()), "b" -> PInt64())
      expectedVal = Annotation(IndexedSeq(1,5,7,2,31415926), 31415926535897L)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}
