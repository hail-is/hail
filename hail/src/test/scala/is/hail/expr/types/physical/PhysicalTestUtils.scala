package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.utils.log
import is.hail.annotations.{Region, ScalaToRegionValue, UnsafeRow}
import is.hail.expr.ir.EmitFunctionBuilder

abstract class PhysicalTestUtils extends HailSuite {
  def copyTestExecutor(sourceType: PType, destType: PType, sourceValue: Any,
    expectCompileErr: Boolean = false, deepCopy: Boolean = false, interpret: Boolean = false, expectedValue: Any = null) {

    val srcRegion = Region()
    val region = Region()

    val srcAddress = sourceType match {
      case x: PSubsetStruct => ScalaToRegionValue(srcRegion, x.ps, sourceValue)
      case _ => ScalaToRegionValue(srcRegion, sourceType, sourceValue)
    }

    if (interpret) {
      try {
        val copyOff = destType.fundamentalType.copyFromAddress(region, sourceType.fundamentalType, srcAddress, deepCopy = deepCopy)
        val copy = UnsafeRow.read(destType, region, copyOff)

        log.info(s"Copied value: ${copy}, Source value: ${sourceValue}")

        if(expectedValue != null) {
          assert(copy == expectedValue)
        } else {
          assert(copy == sourceValue)
        }
        region.clear()
        srcRegion.clear()
      } catch {
        case e: AssertionError =>
          srcRegion.clear()
          region.clear()

          if (expectCompileErr) {
            log.info("OK: Caught expected compile-time error")
            return
          }

          throw new Error(e)
      }

      return
    }
    
    var compileSuccess = false
    val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "not_empty")
    val codeRegion = fb.getCodeParam[Region](1)
    val value = fb.getCodeParam[Long](2)

    try {
      fb.emit(destType.fundamentalType.copyFromType(fb.apply_method, codeRegion, sourceType.fundamentalType, value, deepCopy = deepCopy))
      compileSuccess = true
    } catch {
      case e: AssertionError =>
        srcRegion.clear()
        region.clear()

        if (expectCompileErr) {
          log.info("OK: Caught expected compile-time error")
          return
        }

        throw new Error(e)
    }

    if(compileSuccess && expectCompileErr) {
      region.clear()
      srcRegion.clear()
      throw new Error("Did not receive expected compile time error")
    }

    val f = fb.result()()
    val copyOff = f(region, srcAddress)
    val copy = UnsafeRow.read(destType, region, copyOff)

    log.info(s"Copied value: ${copy}, Source value: ${sourceValue}")

    if(expectedValue != null) {
      assert(copy == expectedValue)
    } else {
      assert(copy == sourceValue)
    }
    region.clear()
    srcRegion.clear()
  }
}