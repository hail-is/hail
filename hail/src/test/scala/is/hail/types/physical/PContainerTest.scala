package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PContainerTest extends PhysicalTestUtils {
  def nullInByte(nElements: Int, missingElement: Int) = {
    IndexedSeq.tabulate(nElements)(i => {
      if (i == missingElement - 1)
        null
      else
        i + 1L
    })
  }

  def testContainsNonZeroBits(sourceType: PArray, data: IndexedSeq[Any]) = {
    val srcRegion = Region(pool=pool)
    val src = ScalaToRegionValue(ctx.stateManager, srcRegion, sourceType, data)

    log.info(s"Testing $data")

    val res = Region.containsNonZeroBits(src + sourceType.lengthHeaderBytes, sourceType.loadLength(src))
    res
  }

  def testContainsNonZeroBitsStaged(sourceType: PArray, data: IndexedSeq[Any]) = {
    val srcRegion = Region(pool=pool)
    val src = ScalaToRegionValue(ctx.stateManager, srcRegion, sourceType, data)

    log.info(s"Testing $data")

    val fb = EmitFunctionBuilder[Long, Boolean](ctx, "not_empty")
    val value = fb.getCodeParam[Long](1)

    fb.emit(Region.containsNonZeroBits(value + sourceType.lengthHeaderBytes, sourceType.loadLength(value).toL))

    val res = fb.result(ctx)(theHailClassLoader)(src)
    res
  }

  def testHasMissingValues(sourceType: PArray, data: IndexedSeq[Any]) = {
    val srcRegion = Region(pool=pool)
    val src = ScalaToRegionValue(ctx.stateManager, srcRegion, sourceType, data)

    log.info(s"\nTesting $data")

    val fb = EmitFunctionBuilder[Long, Boolean](ctx, "not_empty")
    val value = fb.getCodeParam[Long](1)

    fb.emit(sourceType.hasMissingValues(value))

    val res = fb.result(ctx)(theHailClassLoader)(src)
    res
  }

  @Test def checkFirstNonZeroByte() {
    val sourceType = PCanonicalArray(PInt64(false))

    assert(testContainsNonZeroBits(sourceType, nullInByte(0, 0)) == false)

    assert(testContainsNonZeroBits(sourceType, nullInByte(1, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(1, 1)) == true)

    assert(testContainsNonZeroBits(sourceType, nullInByte(8, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(8, 1)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(8, 8)) == true)

    assert(testContainsNonZeroBits(sourceType, nullInByte(32, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(31, 31)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(32, 32)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(33, 33)) == true)

    assert(testContainsNonZeroBits(sourceType, nullInByte(64, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(64, 1)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(64, 32)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(64, 33)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(64, 64)) == true)

    assert(testContainsNonZeroBits(sourceType, nullInByte(68, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(68, 1)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(68, 32)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(68, 33)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(68, 64)) == true)

    assert(testContainsNonZeroBits(sourceType, nullInByte(72, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(72, 1)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(72, 32)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(72, 33)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(72, 64)) == true)

    assert(testContainsNonZeroBits(sourceType, nullInByte(73, 0)) == false)
    assert(testContainsNonZeroBits(sourceType, nullInByte(73, 1)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(73, 32)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(73, 33)) == true)
    assert(testContainsNonZeroBits(sourceType, nullInByte(73, 64)) == true)
  }

  @Test def checkFirstNonZeroByteStaged() {
    val sourceType = PCanonicalArray(PInt64(false))

    assert(testContainsNonZeroBitsStaged(sourceType, nullInByte(32, 0)) == false)
    assert(testContainsNonZeroBitsStaged(sourceType, nullInByte(73, 64)) == true)
  }

  @Test def checkHasMissingValues() {
    val sourceType = PCanonicalArray(PInt64(false))

    assert(testHasMissingValues(sourceType, nullInByte(1, 0)) == false)
    assert(testHasMissingValues(sourceType, nullInByte(1, 1)) == true)
    assert(testHasMissingValues(sourceType, nullInByte(2, 1)) == true)

    for {
      num <- Seq(2, 16, 31, 32, 33, 50, 63, 64, 65, 90, 127, 128, 129)
        missing <- 1 to num
    } assert(testHasMissingValues(sourceType, nullInByte(num, missing)) == true)
  }

  @Test def arrayCopyTest() {
    // Note: can't test where data is null due to ArrayStack.top semantics (ScalaToRegionValue: assert(size_ > 0))
    def runTests(deepCopy: Boolean, interpret: Boolean) {
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt64()), IndexedSeq(1, 2, 3, 4, 5, 6, 7, 8, 9),
        expectCompileError = true, deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt32()), IndexedSeq(1, 2, 3, 4),
        deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt32()), IndexedSeq(1, 2, 3, 4),
        deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt32()), IndexedSeq(1, null, 3, 4),
        deepCopy = deepCopy, interpret = interpret)

      // test upcast
      copyTestExecutor(PCanonicalArray(PInt32(true)), PCanonicalArray(PInt32()), IndexedSeq(1, 2, 3, 4),
        deepCopy = deepCopy, interpret = interpret)

      // test mismatched top-level requiredeness, allowed because by source value address must be present and therefore non-null
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt32(), true), IndexedSeq(1, 2, 3, 4),
        deepCopy = deepCopy, interpret = interpret)

      // downcast disallowed
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt32(true)), IndexedSeq(1, 2, 3, 4),
        deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PInt64())), PCanonicalArray(PCanonicalArray(PInt64(), true)),
        FastSeq(FastSeq(20L), FastSeq(1L), FastSeq(20L,5L,31L,41L), FastSeq(1L,2L,3L)),
        deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PInt64())), PCanonicalArray(PCanonicalArray(PInt64(), true)),
        FastSeq(FastSeq(20L), FastSeq(1L), FastSeq(20L,5L,31L,41L), FastSeq(1L,2L,3L)),
        deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PInt64())), PCanonicalArray(PCanonicalArray(PInt64(true))),
        FastSeq(FastSeq(20L), FastSeq(1L), FastSeq(20L,5L,31L,41L), FastSeq(1L,2L,3L)),
         deepCopy = deepCopy, interpret = interpret)

      // test empty arrays
      copyTestExecutor(PCanonicalArray(PInt32()), PCanonicalArray(PInt32()), FastSeq(),
        deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PInt32(true)), PCanonicalArray(PInt32(true)), FastSeq(),
        deepCopy = deepCopy, interpret = interpret)

      // test missing-only array
      copyTestExecutor(PCanonicalArray(PInt64()), PCanonicalArray(PInt64()),
        FastSeq(null), deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PInt64())), PCanonicalArray(PCanonicalArray(PInt64())),
        FastSeq(FastSeq(null)), deepCopy = deepCopy, interpret = interpret)

      // test 2D arrays
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PInt64())), PCanonicalArray(PCanonicalArray(PInt64())),
        FastSeq(null, FastSeq(null), FastSeq(20L,5L,31L,41L), FastSeq(1L,2L,3L)),
        deepCopy = deepCopy, interpret = interpret)

      // test complex nesting
      val complexNesting = FastSeq(
        FastSeq( FastSeq(20L,30L,31L,41L), FastSeq(20L,22L,31L,43L) ),
        FastSeq( FastSeq(1L,3L,31L,41L), FastSeq(0L,30L,17L,41L) )
      )

      copyTestExecutor(PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64(true), true), true), true), PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64()))),
        complexNesting, deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64(true), true), true)), PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64()))),
        complexNesting, deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64(true), true))), PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64()))),
        complexNesting, deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64(true)))), PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64()))),
        complexNesting, deepCopy = deepCopy, interpret = interpret)
      copyTestExecutor(PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64()))), PCanonicalArray(PCanonicalArray(PCanonicalArray(PInt64()))),
        complexNesting, deepCopy = deepCopy, interpret = interpret)

      val srcType = PCanonicalArray(PCanonicalStruct("a" -> PCanonicalArray(PInt32(true)), "b" -> PInt64()))
      val destType = PCanonicalArray(PCanonicalStruct("a" -> PCanonicalArray(PInt32()), "b" -> PInt64()))
      val expectedVal = IndexedSeq(Annotation(IndexedSeq(1,5,7,2,31415926), 31415926535897L))
      copyTestExecutor(srcType, destType, expectedVal, deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true, false)
    runTests(false, false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }

  @Test def dictCopyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean) {
      copyTestExecutor(PCanonicalDict(PCanonicalString(), PInt32()), PCanonicalDict(PCanonicalString(), PInt32()), Map("test" -> 1),
        deepCopy = deepCopy, interpret = interpret)

      copyTestExecutor(PCanonicalDict(PCanonicalString(true), PInt32(true)), PCanonicalDict(PCanonicalString(), PInt32()), Map("test2" -> 2),
        deepCopy = deepCopy, interpret = interpret)

      copyTestExecutor(PCanonicalDict(PCanonicalString(), PInt32()), PCanonicalDict(PCanonicalString(true), PInt32()), Map("test3" -> 3),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true, false)
    runTests(false, false)
    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }

  @Test def setCopyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean) {
      copyTestExecutor(PCanonicalSet(PCanonicalString(true)), PCanonicalSet(PCanonicalString()), Set("1", "2"),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true, false)
    runTests(false, false)
    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}
