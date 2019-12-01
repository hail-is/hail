package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq, ScalaToRegionValue, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PContainerTest extends HailSuite {
  @Test def checkedConvertFromTest() {
    def nullInByte(nBytes: Int,  nByteWhereNull: Long, indexInByte: Long) = {
      IndexedSeq.tabulate(nBytes*8)(i => {
        if(nByteWhereNull > 0 && i == (nByteWhereNull * 8 - 8 + indexInByte) ) {
          null
        } else {
          i + 1L
        }
      })
    }

    def testIt(sourceType: PArray, destType: PArray, data: IndexedSeq[Any], expectFalse: Boolean) {
      val srcRegion = Region()
      val src = ScalaToRegionValue(srcRegion, sourceType, data)

      log.debug(s"Testing $data")
      log.debug(s"PContainer.nMissingBytes(len) == ${PContainer.nMissingBytes(sourceType.loadLength(src))}")

      val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
      val codeRegion = fb.getArg[Region](1).load()
      val value = fb.getArg[Long](2)

      // checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], oldOffset: Code[Long], otherPT: PContainer, msg: String)
      fb.emit(destType.checkedConvertFrom(fb.apply_method, codeRegion, value, sourceType, "ShouldHaveNoNull"))

      val f = fb.result()()
      val destRegion = Region()
      if(expectFalse) {
        val thrown = intercept[Exception](f(destRegion,src))
        assert(thrown.getMessage == "ShouldHaveNoNull")
      } else {
        f(destRegion,src)
      }
    }

    val sourceType = PArray(PInt64(false))
    val destType = PArray(PInt64(true))

    testIt(sourceType, destType, nullInByte(1, 0, 0), false)
    testIt(sourceType, destType, nullInByte(1, 1, 0), true)
    testIt(sourceType, destType, nullInByte(7, 0, 0), false)
    testIt(sourceType, destType, nullInByte(7, 1, 0), true)
    testIt(sourceType, destType, nullInByte(7, 7, 7), true)
    testIt(sourceType, destType, nullInByte(8, 0, 0), false)
    testIt(sourceType, destType, nullInByte(8, 1, 0), true)
    testIt(sourceType, destType, nullInByte(8, 5, 0), true)
    testIt(sourceType, destType, nullInByte(8, 5, 7), true)
    testIt(sourceType, destType, nullInByte(8, 8, 1), true)
    testIt(sourceType, destType, nullInByte(8, 8, 7), true)
    testIt(sourceType, destType, nullInByte(9, 0, 0), false)
    testIt(sourceType, destType, nullInByte(9, 1, 0), true)
    testIt(sourceType, destType, nullInByte(9, 1, 7), true)
    testIt(sourceType, destType, nullInByte(9, 9, 7), true)
  }

  @Test def testArrayCopy() {
    def testArrayCopy(sourceType: PArray, destType: PArray, sourceValue: IndexedSeq[Any], expectedException: Boolean) {
      val region = Region()
      val srcRegion = Region()

      val srcOffset = ScalaToRegionValue(srcRegion, sourceType, sourceValue)

      val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
      val codeRegion = fb.getArg[Region](1).load()
      val value = fb.getArg[Long](2)

      try {
        fb.emit(destType.copyFromType(fb, codeRegion, sourceType, value))

        val f = fb.result()()
        val copyOff = f(region, srcOffset)
        val copy = SafeIndexedSeq(destType, region, copyOff)
        assert(copy == sourceValue)
        log.debug(s"Copied value: ${copy}")
      } catch {
        case e: Throwable => {
          if(expectedException) {
            log.debug(s"Found expected exception: ${e.getMessage}")
          } else {
            throw new Error(e)
          }
        }
      }
    }

    testArrayCopy(PArray(PInt32()), PArray(PInt32()), IndexedSeq(1, 2, 3, 4, 5, 6, 7, 8, null), false)
    // Such tests, where array is null are not currently possible due to ArrayStack.top semantics (ScalaToRegionValue)
    // testArrayCopy(PArray(PInt32()), PArray(PInt32()), null, false)
    testArrayCopy(PArray(PInt32()), PArray(PInt32()), FastIndexedSeq(), false)
    testArrayCopy(PArray(PInt32(true)), PArray(PInt32()), IndexedSeq(1, 2, 3, 4), false)
    testArrayCopy(PArray(PInt32(true)), PArray(PInt32(true)), IndexedSeq(1, 2, 3, 4), false)
    testArrayCopy(PArray(PInt32(false)), PArray(PInt32(true)), IndexedSeq(1, 2, 3, 4), false)
    testArrayCopy(PArray(PInt32(false)), PArray(PInt32(true)), IndexedSeq(null, 2, 3, 4), true)
    testArrayCopy(PArray(PInt32()), PArray(PInt64()), IndexedSeq(1, 2, 3, 4, 5, 6, 7, 8, 9), true)

    testArrayCopy(PArray(PArray(PInt64(false))), PArray(PArray(PInt64(true))),
      FastIndexedSeq(null, FastIndexedSeq(20L,null,31L,41L), FastIndexedSeq(null,null,null,null)), true)
    testArrayCopy(PArray(PArray(PInt64(true), true)), PArray(PArray(PInt64(false))),
      FastIndexedSeq(FastIndexedSeq(1L), FastIndexedSeq(20L,5L,31L,41L), FastIndexedSeq(1L,2L,3L)), false)
  }
}
