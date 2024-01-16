package is.hail.types.physical
import is.hail.annotations.Annotation

import org.testng.annotations.Test

class PBaseStructSuite extends PhysicalTestUtils {
  @Test def testStructCopy(): Unit = {
    def runTests(deepCopy: Boolean, interpret: Boolean = false): Unit = {
      copyTestExecutor(
        PCanonicalStruct(),
        PCanonicalStruct(),
        Annotation(),
        deepCopy = deepCopy,
        interpret = interpret,
      )
      copyTestExecutor(
        PCanonicalStruct("a" -> PInt64()),
        PCanonicalStruct("a" -> PInt64()),
        Annotation(12L),
        deepCopy = deepCopy,
        interpret = interpret,
      )
      copyTestExecutor(
        PCanonicalStruct("a" -> PInt64()),
        PCanonicalStruct("a" -> PInt64()),
        Annotation(null),
        deepCopy = deepCopy,
        interpret = interpret,
      )
      copyTestExecutor(
        PCanonicalStruct("a" -> PInt64(true)),
        PCanonicalStruct("a" -> PInt64(true)),
        Annotation(11L),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalStruct("a" -> PInt64(false)),
        PCanonicalStruct("a" -> PInt64(true)),
        Annotation(3L),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      var srcType = PCanonicalStruct(
        "a" -> PInt64(true),
        "b" -> PInt32(true),
        "c" -> PFloat64(true),
        "d" -> PFloat32(true),
        "e" -> PBoolean(true),
      )
      var destType = PCanonicalStruct(
        "a" -> PInt64(true),
        "b" -> PInt32(true),
        "c" -> PFloat64(true),
        "d" -> PFloat32(true),
        "e" -> PBoolean(true),
      )
      var expectedVal = Annotation(13L, 12, 13.0, 10.0f, true)

      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PCanonicalStruct("a" -> srcType, "c" -> PFloat32())
      destType = PCanonicalStruct("a" -> destType, "c" -> PFloat32())
      var nestedExpectedVal = Annotation(expectedVal, 13.0f)
      copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy,
        interpret = interpret)

      srcType = PCanonicalStruct(
        "a" -> PInt64(),
        "b" -> PInt32(),
        "c" -> PFloat64(),
        "d" -> PFloat32(),
        "e" -> PBoolean(),
      )
      destType = PCanonicalStruct(
        "a" -> PInt64(),
        "b" -> PInt32(),
        "c" -> PFloat64(),
        "d" -> PFloat32(),
        "e" -> PBoolean(),
      )
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PCanonicalStruct("a" -> srcType, "b" -> PFloat32())
      destType = PCanonicalStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 14.0f)
      copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy,
        interpret = interpret)

      srcType = PCanonicalStruct(
        "a" -> PInt64(),
        "b" -> PInt32(true),
        "c" -> PFloat64(),
        "d" -> PFloat32(),
        "e" -> PBoolean(),
      )
      destType = PCanonicalStruct(
        "a" -> PInt64(),
        "b" -> PInt32(),
        "c" -> PFloat64(),
        "d" -> PFloat32(),
        "e" -> PBoolean(),
      )
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PCanonicalStruct("a" -> srcType, "b" -> PFloat32())
      destType = PCanonicalStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(Annotation(13L, 12, 13.0, 10.0f, true), 15.0f)
      copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy,
        interpret = interpret)

      srcType = PCanonicalStruct(
        "a" -> PInt64(),
        "b" -> PInt32(true),
        "c" -> PFloat64(),
        "d" -> PFloat32(),
        "e" -> PBoolean(),
      )
      destType = PCanonicalStruct(
        "a" -> PInt64(),
        "b" -> PInt32(),
        "c" -> PFloat64(true),
        "d" -> PFloat32(),
        "e" -> PBoolean(),
      )
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy)

      srcType = PCanonicalStruct("a" -> srcType, "b" -> PFloat32())
      destType = PCanonicalStruct("a" -> destType, "b" -> PFloat32())
      nestedExpectedVal = Annotation(expectedVal, 13f)
      copyTestExecutor(srcType, destType, nestedExpectedVal, deepCopy = deepCopy)

      srcType = PCanonicalStruct("a" -> PCanonicalArray(PInt32(true)), "b" -> PInt64())
      destType = PCanonicalStruct("a" -> PCanonicalArray(PInt32()), "b" -> PInt64())
      expectedVal = Annotation(IndexedSeq(1, 5, 7, 2, 31415926), 31415926535897L)
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      expectedVal = Annotation(null, 31415926535897L)
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)
      expectedVal = Annotation(null, null)
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PCanonicalStruct("a" -> PCanonicalArray(PInt32(true)), "b" -> PInt64())
      destType = PCanonicalStruct("a" -> PCanonicalArray(PInt32(), true), "b" -> PInt64())
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret,
        expectRuntimeError = true)

      srcType = PCanonicalStruct(
        "a" -> PCanonicalArray(PCanonicalArray(PCanonicalStruct("a" -> PInt32(true)))),
        "b" -> PInt64(),
      )
      destType = PCanonicalStruct(
        "a" -> PCanonicalArray(PCanonicalArray(PCanonicalStruct("a" -> PInt32(true)))),
        "b" -> PInt64(),
      )
      expectedVal = Annotation(IndexedSeq(null, IndexedSeq(null, Annotation(1))), 31415926535897L)
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)

      srcType = PCanonicalStruct(
        true,
        "foo" -> PCanonicalStruct("bar" -> PCanonicalArray(PInt32(true), true)),
      )
      destType = PCanonicalStruct(
        false,
        "foo" -> PCanonicalStruct("bar" -> PCanonicalArray(PInt32(false), false)),
      )
      expectedVal = Annotation(Annotation(IndexedSeq(1, 2, 3)))
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }

  @Test def tupleCopyTests(): Unit = {
    def runTests(deepCopy: Boolean, interpret: Boolean = false): Unit = {
      copyTestExecutor(
        PCanonicalTuple(false, PCanonicalString(true), PCanonicalString(true)),
        PCanonicalTuple(false, PCanonicalString(), PCanonicalString()),
        Annotation("1", "2"),
        deepCopy = deepCopy,
        interpret = interpret,
      )
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}
