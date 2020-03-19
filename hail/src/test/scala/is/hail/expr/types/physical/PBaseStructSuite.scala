package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class PBaseStructSuite extends HailSuite {
  @Test def testStructCopy() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PStruct(), PStruct(), Annotation(),
        deepCopy = deepCopy, interpret = interpret)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64()), PStruct("a" -> PInt64()), Annotation(12L),
        deepCopy = deepCopy, interpret = interpret)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64()), PStruct("a" -> PInt64()), Annotation(null),
        deepCopy = deepCopy, interpret = interpret)
      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(true)), PStruct("a" -> PInt64(true)), Annotation(11L),
        deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PStruct("a" -> PInt64(false)), PStruct("a" -> PInt64(true)), Annotation(3L),
        expectCompileErr = true, deepCopy = deepCopy, interpret = interpret)

      var srcType = PStruct("a" -> PInt64(true), "b" -> PInt32(true), "c" -> PFloat64(true), "d" -> PFloat32(true), "e" -> PBoolean(true))
      var destType = PStruct("a" -> PInt64(true), "b" -> PInt32(true), "c" -> PFloat64(true), "d" -> PFloat32(true), "e" -> PBoolean(true))
      var expectedVal = Annotation(13L, 12, 13.0, 10.0F, true)

      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> srcType, "c" -> PFloat32())
      destType = PStruct("a" -> destType, "c" -> PFloat32())
      var nestedExpectedVal = Annotation(expectedVal, 13.0F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      destType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> srcType, "b" -> PFloat32())
      destType = PStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 14.0F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> PInt64(), "b" -> PInt32(true), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      destType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> srcType, "b" -> PFloat32())
      destType = PStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(Annotation(13L, 12, 13.0, 10.0F, true), 15.0F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> PInt64(), "b" -> PInt32(true), "c" -> PFloat64(), "d" -> PFloat32(), "e" -> PBoolean())
      destType = PStruct("a" -> PInt64(), "b" -> PInt32(), "c" -> PFloat64(true), "d" -> PFloat32(), "e" -> PBoolean())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, expectCompileErr = true, deepCopy = deepCopy)

      srcType = PStruct("a" -> srcType, "b" -> PFloat32())
      destType = PStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 13F)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, nestedExpectedVal, expectCompileErr = true, deepCopy = deepCopy)

      srcType = PStruct("a" -> PArray(PInt32(true)), "b" -> PInt64())
      destType = PStruct("a" -> PArray(PInt32()), "b" -> PInt64())
      expectedVal = Annotation(IndexedSeq(1,5,7,2,31415926), 31415926535897L)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      expectedVal = Annotation(null, 31415926535897L)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)
      expectedVal = Annotation(null, null)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> PArray(PInt32(true)), "b" -> PInt64())
      destType = PStruct("a" -> PArray(PInt32(), true), "b" -> PInt64())
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, expectCompileErr = true, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct("a" -> PArray(PArray(PStruct("a" -> PInt32(true)))), "b" -> PInt64())
      destType = PStruct("a" -> PArray(PArray(PStruct("a" -> PInt32(true)))), "b" -> PInt64())
      expectedVal = Annotation(IndexedSeq(null, IndexedSeq(null, Annotation(1))), 31415926535897L)
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PStruct(true, "foo" -> PStruct("bar" -> PArray(PInt32(true), true)))
      destType = PStruct(false, "foo" -> PStruct("bar" -> PArray(PInt32(false), false)))
      expectedVal = Annotation(Annotation(IndexedSeq(1, 2, 3)))
      PhysicalTestUtils.copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }

  @Test def tupleCopyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PTuple(PString(true), PString(true)), PTuple(PString(), PString()), Annotation("1", "2"),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}
