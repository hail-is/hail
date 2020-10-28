package is.hail.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import org.testng.annotations.Test

class PNDArraySuite extends PhysicalTestUtils {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      copyTestExecutor(PCanonicalNDArray(PInt64(true), 1), PCanonicalNDArray(PInt64(true), 1), Annotation(Annotation(1L), IndexedSeq(4L,5L,6L)),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }

  @Test def testRefCounted(): Unit = {
    val nd1 = new PReferenceCountedNDArray(PInt32Required, 1)

    val region = Region()
    val fb = EmitFunctionBuilder[Region, Long](ctx, "ref_count_test")
    val codeRegion = fb.getCodeParam[Region](1)

    try {
      fb.emitWithBuilder{ cb =>
        val shapeAddress = cb.newLocal[Long]("shape_address")
        val ndAddress = cb.newLocal[Long]("nd_address")

        val shapeSeq = IndexedSeq(const(3L))
        val shapeMaker = nd1.makeShapeBuilder(shapeSeq)
        val shapeSRVB = new StagedRegionValueBuilder(cb.emb, nd1.shape.pType, codeRegion)
        cb.append(shapeMaker(shapeSRVB))
        cb.assign(shapeAddress, shapeSRVB.end())
        val shapePCode = PCode(nd1.shape.pType, shapeAddress).asBaseStruct
        val shapePValue = shapePCode.memoize(cb, "testRefCounted_shape")

        cb.assign(ndAddress, nd1.allocateAndInitialize(cb, codeRegion, shapePValue, shapePValue))
        ndAddress
      }
    } catch {
      case e: AssertionError =>
        region.clear()
        throw e
    }

    val f = fb.result()()
    val result1 = f(region)

    // Simple clearing test

    assert(region.memory.listNDArrayRefs().size == 1)
    assert(region.memory.listNDArrayRefs()(0) == result1)

    region.clear()
    assert(region.memory.listNDArrayRefs().size == 0)
  }
}
