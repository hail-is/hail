package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PContainerTest extends HailSuite {
  @Test def checkedConvertFromTest() {
    def nullInByte(nElements: Int, missingElement: Int) = {
      IndexedSeq.tabulate(nElements)(i => {
        if(i == missingElement - 1) {
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

      val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
      val codeRegion = fb.getArg[Region](1).load()
      val value = fb.getArg[Long](2)

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

    // 1 byte
    testIt(sourceType, destType, nullInByte(1, 0), false)
    testIt(sourceType, destType, nullInByte(1, 1), true)
    testIt(sourceType, destType, nullInByte(5, 5), true)

    // 1 full byte
    testIt(sourceType, destType, nullInByte(8, 0), false)
    testIt(sourceType, destType, nullInByte(8, 1), true)
    testIt(sourceType, destType, nullInByte(8, 5), true)
    testIt(sourceType, destType, nullInByte(8, 8), true)

    // 1 byte + remainder
    testIt(sourceType, destType, nullInByte(11, 0), false)
    testIt(sourceType, destType, nullInByte(13, 13), true)
    testIt(sourceType, destType, nullInByte(13, 9), true)
    testIt(sourceType, destType, nullInByte(13, 8), true)

    // 1 Long
    testIt(sourceType, destType, nullInByte(64, 0), false)
    testIt(sourceType, destType, nullInByte(64, 1), true)
    testIt(sourceType, destType, nullInByte(64, 64), true)

    // 1 Long + remainder
    testIt(sourceType, destType, nullInByte(67, 0), false)
    testIt(sourceType, destType, nullInByte(67, 67), true)
    testIt(sourceType, destType, nullInByte(67, 65), true)
    testIt(sourceType, destType, nullInByte(67, 64), true)

    // 1 Long + 1 byte + remainder
    testIt(sourceType, destType, nullInByte(79, 8), true)

  }
}
