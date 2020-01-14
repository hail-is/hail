package is.hail.expr.types.physical
import is.hail.utils.log
import is.hail.annotations.{Region, SafeIndexedSeq, SafeRow, ScalaToRegionValue, UnsafeRow}
import is.hail.expr.ir.EmitFunctionBuilder

object PhysicalTestUtils {
  def copyTestExecutor(sourceType: PType, destType: PType, sourceValue: Any,
    expectCompileErr: Boolean = false, expectRuntimeErr: Boolean = false,
    allowDowncast: Boolean = false, forceDeep: Boolean = false, interpret: Boolean = false) {
    val region = Region()
    val srcRegion = Region()

    val srcAddress = ScalaToRegionValue(srcRegion, sourceType, sourceValue)

    var runtimeSuccess = false
    var compileSuccess = false
    if(!interpret) {
      val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
      val codeRegion = fb.getArg[Region](1).load()
      val value = fb.getArg[Long](2)

      try {
        fb.emit(destType.copyFromType(fb.apply_method, codeRegion, sourceType, value,
          allowDowncast = allowDowncast, forceDeep = forceDeep))
        compileSuccess = true
      } catch {
        case e: Throwable => {
          region.clear()
          srcRegion.clear()
          if(expectCompileErr) {
            log.info("OK: Caught expected compile-time error")
            return assert(true)
          }

          throw new Error(e)
        }
      }

      if(compileSuccess && expectCompileErr) {
        region.clear()
        srcRegion.clear()
        throw new Error("Did not receive expected compile time error")
      }

      try {
        val f = fb.result()()
        val copyOff = f(region, srcAddress)
        val copy = UnsafeRow.read(destType, region, copyOff)

        log.info(s"Copied value: ${copy}, Source value: ${sourceValue}")
        assert(copy == sourceValue)
        runtimeSuccess = true
        region.clear()
        srcRegion.clear()
      } catch {
        case e: Throwable => {
          region.clear()
          srcRegion.clear()
          if(expectRuntimeErr) {
            log.info(s"OK: Found expected runtime failure: ${e.getMessage}")
          } else {
            throw new Error(e)
          }
        }
      }
    } else {
      try {
        val copyOff = destType.copyFromType(region, sourceType, srcAddress,
          allowDowncast = allowDowncast, forceDeep = forceDeep)
        val copy = UnsafeRow.read(destType, region, copyOff)

        log.info(s"Copied value: ${copy}, Source value: ${sourceValue}")
        assert(copy == sourceValue)
        runtimeSuccess = true
        region.clear()
        srcRegion.clear()
      } catch {
        case e: Throwable => {
          region.clear()
          srcRegion.clear()
          if(expectRuntimeErr) {
            log.info(s"OK: Found expected runtime failure: ${e.getMessage}")
          } else if(expectCompileErr) {
            log.info(s"OK: In interpreted mode, received runtime error where expected compile time error: ${e.getMessage}")
          } else {
            throw new Error(e)
          }
        }
      }
    }

    if(runtimeSuccess && expectRuntimeErr) {
      throw new Error("Did not receive expected runtime error")
    }
  }
}