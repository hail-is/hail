package is.hail.expr.types.physical
import is.hail.utils.log
import is.hail.annotations.{Region, SafeIndexedSeq, SafeRow, ScalaToRegionValue, UnsafeRow}
import is.hail.expr.ir.EmitFunctionBuilder
import org.scalatest.testng.TestNGSuite

object PhysicalTestUtils extends TestNGSuite {
  def copyTestExecutor(sourceType: PType, destType: PType, sourceValue: Any,
    expectCompileErr: Boolean = false, deepCopy: Boolean = false, interpret: Boolean = false) {

    val srcRegion = Region()
    val region = Region()

    val srcAddress = ScalaToRegionValue(srcRegion, sourceType, sourceValue)

    if(interpret) {
      try {
        val copyOff = destType.fundamentalType.copyFromType(region, sourceType.fundamentalType, srcAddress, deepCopy = deepCopy)
        val copy = UnsafeRow.read(destType, region, copyOff)

        log.info(s"Copied value: ${copy}, Source value: ${sourceValue}")
        assert(copy == sourceValue)
        region.clear()
        srcRegion.clear()
      } catch {
        case e: AssertionError => {
          srcRegion.clear()
          region.clear()

          if(expectCompileErr) {
            log.info("OK: Caught expected compile-time error")
            return
          }

          throw new Error(e)
        }
      }

      return
    }
    
    var compileSuccess = false
    val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
    val codeRegion = fb.getArg[Region](1)
    val value = fb.getArg[Long](2)

    try {
      fb.emit(destType.fundamentalType.copyFromType(fb.apply_method, codeRegion, sourceType.fundamentalType, value, deepCopy = deepCopy))
      compileSuccess = true
    } catch {
      case e: AssertionError => {
        srcRegion.clear()
        region.clear()

        if(expectCompileErr) {
          log.info("OK: Caught expected compile-time error")
          return
        }

        throw new Error(e)
      }
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
    assert(copy == sourceValue)
    region.clear()
    srcRegion.clear()
  }
}