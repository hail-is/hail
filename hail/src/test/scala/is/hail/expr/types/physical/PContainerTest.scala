package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder}
import is.hail.utils._
import org.testng.annotations.Test

class PContainerTest extends HailSuite {
  def nullInByte(nElements: Int, missingElement: Int) = {
    IndexedSeq.tabulate(nElements)(i => {
      if (i == missingElement - 1)
        null
      else
        i + 1L
    })
  }

  def testConvert(sourceType: PArray, destType: PArray, data: IndexedSeq[Any], expectFalse: Boolean) {
    val srcRegion = Region()
    val src = ScalaToRegionValue(srcRegion, sourceType, data)

    log.info(s"Testing $data")

    val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
    val codeRegion = fb.getArg[Region](1).load()
    val value = fb.getArg[Long](2)

    fb.emit(destType.checkedConvertFrom(fb.apply_method, codeRegion, value, sourceType, "ShouldHaveNoNull"))

    val f = fb.result()()
    val destRegion = Region()
    if (expectFalse) {
      val thrown = intercept[Exception](f(destRegion,src))
      assert(thrown.getMessage == "ShouldHaveNoNull")
    } else
      f(destRegion,src)
  }

  def testContainsNonZeroBits(sourceType: PArray, data: IndexedSeq[Any]) = {
    val srcRegion = Region()
    val src = ScalaToRegionValue(srcRegion, sourceType, data)

    log.info(s"Testing $data")

    val res = Region.containsNonZeroBits(src + sourceType.lengthHeaderBytes, sourceType.loadLength(src))
    res
  }

  def testContainsNonZeroBitsStaged(sourceType: PArray, data: IndexedSeq[Any]) = {
    val srcRegion = Region()
    val src = ScalaToRegionValue(srcRegion, sourceType, data)

    log.info(s"Testing $data")

    val fb = EmitFunctionBuilder[Long, Boolean]("not_empty")
    val value = fb.getArg[Long](1)

    fb.emit(Region.containsNonZeroBits(value + sourceType.lengthHeaderBytes, sourceType.loadLength(value).toL))

    val res = fb.result()()(src)
    res
  }

  def testHasMissingValues(sourceType: PArray, data: IndexedSeq[Any]) = {
    val srcRegion = Region()
    val src = ScalaToRegionValue(srcRegion, sourceType, data)

    log.info(s"\nTesting $data")

    val fb = EmitFunctionBuilder[Long, Boolean]("not_empty")
    val value = fb.getArg[Long](1)

    fb.emit(sourceType.hasMissingValues(value))

    val res = fb.result()()(src)
    res
  }

  @Test def checkFirstNonZeroByte() {
    val sourceType = PArray(PInt64(false))

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
    val sourceType = PArray(PInt64(false))

    assert(testContainsNonZeroBitsStaged(sourceType, nullInByte(32, 0)) == false)
    assert(testContainsNonZeroBitsStaged(sourceType, nullInByte(73, 64)) == true)
  }

  @Test def checkHasMissingValues() {
    val sourceType = PArray(PInt64(false))

    assert(testHasMissingValues(sourceType, nullInByte(1, 0)) == false)
    assert(testHasMissingValues(sourceType, nullInByte(1, 1)) == true)
  }

  @Test def checkedConvertFromTest() {
    val sourceType = PArray(PInt64(false))
    val destType = PArray(PInt64(true))

    testConvert(sourceType, destType, nullInByte(0, 0), false)

    // 1 byte
    testConvert(sourceType, destType, nullInByte(1, 0), false)
    testConvert(sourceType, destType, nullInByte(1, 1), true)
    testConvert(sourceType, destType, nullInByte(5, 5), true)

    // 1 full byte
    testConvert(sourceType, destType, nullInByte(8, 0), false)
    testConvert(sourceType, destType, nullInByte(8, 1), true)
    testConvert(sourceType, destType, nullInByte(8, 5), true)
    testConvert(sourceType, destType, nullInByte(8, 8), true)

    // 1 byte + remainder
    testConvert(sourceType, destType, nullInByte(11, 0), false)
    testConvert(sourceType, destType, nullInByte(13, 13), true)
    testConvert(sourceType, destType, nullInByte(13, 9), true)
    testConvert(sourceType, destType, nullInByte(13, 8), true)

    // 1 Long
    testConvert(sourceType, destType, nullInByte(64, 0), false)
    testConvert(sourceType, destType, nullInByte(64, 1), true)
    testConvert(sourceType, destType, nullInByte(64, 64), true)

    // 1 Long + remainder
    testConvert(sourceType, destType, nullInByte(67, 0), false)
    testConvert(sourceType, destType, nullInByte(67, 67), true)
    testConvert(sourceType, destType, nullInByte(67, 65), true)
    testConvert(sourceType, destType, nullInByte(67, 64), true)

    // 1 Long + 1 byte + remainder
    testConvert(sourceType, destType, nullInByte(79, 0), false)
    testConvert(sourceType, destType, nullInByte(79, 72), true)
    testConvert(sourceType, destType, nullInByte(79, 8), true)
  }
}