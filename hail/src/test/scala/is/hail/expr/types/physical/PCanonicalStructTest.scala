package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, SafeRow, ScalaToRegionValue, UnsafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PCanonicalStructTest extends HailSuite {
  @Test def testStructCopy() {
      def structCopyExecutor(sourceType: PStruct, destType: PStruct, sourceValue: Any,
        expectCompileErr: Boolean = false, expectRuntimeErr: Boolean = false,
        allowDowncast: Boolean = false, forceDeep: Boolean = false) {
        val region = Region()
        val srcRegion = Region()
        println(s"Source type ${sourceType}")
        println(s"Source value ${sourceValue}")
        val srcOffset = ScalaToRegionValue(srcRegion, sourceType, sourceValue)
        println("past srcOffset")
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

        println("past compile time success")

        var runtimeSuccess = false
        try {
          val f = fb.result()()
          println("past fb.result()()")
          val copyOff = f(region, srcOffset)
          println("Past copy oFF")
          val copy = Region.loadInt(destType.loadField(copyOff, 0))

          println(s"COPY ${copy}")



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
        structCopyExecutor(PStruct("a" -> PInt32()), PStruct("a" -> PInt32()), Annotation(1),
          forceDeep = forceDeep)
      }

      runTests(true)
      runTests(false)
  }
}
