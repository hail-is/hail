package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq, ScalaToRegionValue, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder}
import is.hail.utils._
import org.testng.annotations.Test

class PBaseStructTest extends HailSuite {
  @Test def testArrayCopy() {
    def testArrayCopy(sourceType: PArray, destType: PArray, sourceValue: IndexedSeq[Any],
      expectCompileErr: Boolean = false, expectRuntimeErr: Boolean = false,
      allowDowncast: Boolean = false, forceDeep: Boolean = false) {
      val region = Region()
      val srcRegion = Region()

      val srcOffset = ScalaToRegionValue(srcRegion, sourceType, sourceValue)

      val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
      val codeRegion = fb.getArg[Region](1).load()
      val value = fb.getArg[Long](2)

      var compileSuccess = false
      try {
        fb.emit(destType.copyFromType(fb.apply_method, codeRegion, sourceType, value,
          allowDowncast = allowDowncast, forceDeep = forceDeep))
        compileSuccess = true
      } catch {
        case e: Throwable => {
          if(expectCompileErr) {
            println(s"Found expected compile time error: ${e.getMessage}")
            return assert(true)
          } else {
            throw new Error(e)
          }
        }
      }

      if(compileSuccess && expectCompileErr) {
        throw new Error("Did not receive expected compile time error")
      }

      var runtimeSuccess = false
      try {
        val f = fb.result()()
        val copyOff = f(region, srcOffset)
        val copy = SafeIndexedSeq(destType, region, copyOff)

        println(s"Copied value: ${copy}, Source value: ${sourceValue}")
        assert(copy == sourceValue)
        runtimeSuccess = true
      } catch {
        case e: Throwable => {
          if(expectRuntimeErr) {
            println(s"Found expected failure: ${e.getMessage}")
          } else {
            throw new Error(e)
          }
        }
      }

      if(runtimeSuccess && expectRuntimeErr) {
        throw new Error("Did not receive expected runtime error")
      }
    }

    // Note: can't test where data is null due to ArrayStack.top semantics (ScalaToRegionValue: assert(size_ > 0))

    def runTests(forceDeep: Boolean) {
      testArrayCopy(PArray(PInt32()), PArray(PInt64()), IndexedSeq(1, 2, 3, 4, 5, 6, 7, 8, 9),
        expectCompileErr = true, forceDeep = forceDeep)

      testArrayCopy(PArray(PInt32()), PArray(PInt32()), IndexedSeq(1, 2, 3, 4),
        forceDeep = forceDeep)
      testArrayCopy(PArray(PInt32()), PArray(PInt32()), IndexedSeq(1, 2, 3, 4),
        forceDeep = forceDeep)
      testArrayCopy(PArray(PInt32()), PArray(PInt32()), IndexedSeq(1, null, 3, 4),
        forceDeep = forceDeep)

      // test upcast
      testArrayCopy(PArray(PInt32(true)), PArray(PInt32()), IndexedSeq(1, 2, 3, 4),
        forceDeep = forceDeep)

      // test mismatched top-level requiredeness
      testArrayCopy(PArray(PInt32()), PArray(PInt32(), true), IndexedSeq(1, 2, 3, 4),
        forceDeep = forceDeep)

      // test downcast
      testArrayCopy(PArray(PInt32()), PArray(PInt32(true)), IndexedSeq(1, 2, 3, 4),
        expectRuntimeErr = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PInt32()), PArray(PInt32(true)), IndexedSeq(1, 2, 3, 4),
        allowDowncast = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PInt32()), PArray(PInt32(true)), IndexedSeq(1, null, 3, 4),
        expectRuntimeErr = true, allowDowncast = true, forceDeep = forceDeep)

      // test empty arrays
      testArrayCopy(PArray(PInt32()), PArray(PInt32()), FastIndexedSeq(),
        forceDeep = forceDeep)
      testArrayCopy(PArray(PInt32(true)), PArray(PInt32(true)), FastIndexedSeq(),
        forceDeep = forceDeep)

      // test missing-only array
      testArrayCopy(PArray(PInt64()), PArray(PInt64()),
        FastIndexedSeq(null), forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PInt64())), PArray(PArray(PInt64())),
        FastIndexedSeq(FastIndexedSeq(null)), forceDeep = forceDeep)

      // test 2D arrays
      testArrayCopy(PArray(PArray(PInt64())), PArray(PArray(PInt64())),
        FastIndexedSeq(null, FastIndexedSeq(null), FastIndexedSeq(20L,5L,31L,41L), FastIndexedSeq(1L,2L,3L)),
        forceDeep = forceDeep)

      // test 2D array with missingness
      testArrayCopy(PArray(PArray(PInt64())), PArray(PArray(PInt64(), true)),
        FastIndexedSeq(FastIndexedSeq(20L), FastIndexedSeq(1L), FastIndexedSeq(20L,5L,31L,41L), FastIndexedSeq(1L,2L,3L)),
        allowDowncast = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PInt64())), PArray(PArray(PInt64(), true)),
        FastIndexedSeq(null, FastIndexedSeq(1L), FastIndexedSeq(20L,5L,31L,41L), FastIndexedSeq(1L,2L,3L)),
        allowDowncast = true, expectRuntimeErr = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PInt64())), PArray(PArray(PInt64(true))),
        FastIndexedSeq(FastIndexedSeq(99L), FastIndexedSeq(20L,5L,31L,41L), FastIndexedSeq(1L,2L,3L)),
        allowDowncast = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PInt64())), PArray(PArray(PInt64(true))),
        FastIndexedSeq(FastIndexedSeq(99L), FastIndexedSeq(20L,3L,31L,41L), FastIndexedSeq(1L,2L, null)),
        allowDowncast = true, expectRuntimeErr = true, forceDeep = forceDeep)

      // test complex nesting
      val complexNesting = FastIndexedSeq(
        FastIndexedSeq( FastIndexedSeq(20L,30L,31L,41L), FastIndexedSeq(20L,22L,31L,43L) ),
        FastIndexedSeq( FastIndexedSeq(1L,3L,31L,41L), FastIndexedSeq(0L,30L,17L,41L) )
      )

      testArrayCopy(PArray(PArray(PArray(PInt64(true), true), true), true), PArray(PArray(PArray(PInt64()))),
        complexNesting, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64(true), true), true)), PArray(PArray(PArray(PInt64()))),
        complexNesting, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64(true), true))), PArray(PArray(PArray(PInt64()))),
        complexNesting, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64(true)))), PArray(PArray(PArray(PInt64()))),
        complexNesting, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64()))), PArray(PArray(PArray(PInt64()))),
        complexNesting, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64()))), PArray(PArray(PArray(PInt64(true)))),
        complexNesting, allowDowncast = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64()))), PArray(PArray(PArray(PInt64(true), true))),
        complexNesting, allowDowncast = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64()))), PArray(PArray(PArray(PInt64(true), true), true)),
        complexNesting, allowDowncast = true, forceDeep = forceDeep)
      testArrayCopy(PArray(PArray(PArray(PInt64()))), PArray(PArray(PArray(PInt64(true), true), true), true),
        complexNesting, allowDowncast = true, forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}
