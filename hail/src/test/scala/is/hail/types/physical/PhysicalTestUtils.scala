package is.hail.types.physical

import is.hail.HailSuite
import is.hail.utils.{HailException, log}
import is.hail.annotations.{Region, ScalaToRegionValue, UnsafeRow}
import is.hail.expr.ir.EmitFunctionBuilder

abstract class PhysicalTestUtils extends HailSuite {
  def copyTestExecutor(sourceType: PType, destType: PType, sourceValue: Any,
    expectCompileError: Boolean = false, expectRuntimeError: Boolean = false,
    deepCopy: Boolean = false, interpret: Boolean = false, expectedValue: Any = null) {

    val srcRegion = Region(pool=pool)
    val region = Region(pool=pool)

    val srcAddress = sourceType match {
      case x: PSubsetStruct => ScalaToRegionValue(ctx.stateManager, srcRegion, x.ps, sourceValue)
      case _ => ScalaToRegionValue(ctx.stateManager, srcRegion, sourceType, sourceValue)
    }

    if (interpret) {
      try {
        val copyOff = destType.copyFromAddress(ctx.stateManager, region, sourceType, srcAddress, deepCopy = deepCopy)
        val copy = UnsafeRow.read(destType, copyOff)

        log.info(s"Copied value: ${copy}, Source value: ${sourceValue}")

        if(expectedValue != null) {
          assert(copy == expectedValue)
        } else {
          assert(copy == sourceValue)
        }
        region.clear()
        srcRegion.clear()
      } catch {
        case e: Throwable =>
          srcRegion.clear()
          region.clear()

          if (expectCompileError || expectRuntimeError) {
            log.info("OK: Caught expected compile-time error")
            return
          }

          throw e
      }

      return
    }

    var compileSuccess = false
    val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "not_empty")
    val codeRegion = fb.getCodeParam[Region](1)
    val value = fb.getCodeParam[Long](2)

    try {
      fb.emitWithBuilder(cb => destType.store(cb, codeRegion, sourceType.loadCheapSCode(cb, value), deepCopy = deepCopy))
      compileSuccess = true
    } catch {
      case e: Throwable =>
        srcRegion.clear()
        region.clear()

        if (expectCompileError) {
          log.info("OK: Caught expected compile-time error")
          return
        }

        throw e
    }

    if (compileSuccess && expectCompileError) {
      region.clear()
      srcRegion.clear()
      throw new Error("Did not receive expected compile time error")
    }

    val copy = try {
      val f = fb.result()(theHailClassLoader)
      val copyOff = f(region, srcAddress)
      UnsafeRow.read(destType, copyOff)
    } catch {
      case e: HailException =>
        if (expectRuntimeError) {
          log.info("OK: Caught expected compile-time error")
          return
        }

        throw e
    }

    log.info(s"Copied value: ${ copy }, Source value: ${ sourceValue }")

    if(expectedValue != null) {
      assert(copy == expectedValue)
    } else {
      assert(copy == sourceValue)
    }
    region.clear()
    srcRegion.clear()
  }
}
