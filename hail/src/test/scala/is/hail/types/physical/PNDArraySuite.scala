package is.hail.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import org.testng.annotations.Test
import is.hail.utils._

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
    val nd = new PReferenceCountedNDArray(PInt32Required, 1)

    val region1 = Region()
    val region2 = Region()
    val region3 = Region()
    val fb = EmitFunctionBuilder[Region, Region, Region, Long](ctx, "ref_count_test")
    val codeRegion1 = fb.getCodeParam[Region](1)
    val codeRegion2 = fb.getCodeParam[Region](2)
    val codeRegion3 = fb.getCodeParam[Region](3)

    try {
      fb.emitWithBuilder{ cb =>
        val shapeAddress = cb.newLocal[Long]("shape_address")
        val ndAddress1 = cb.newLocal[Long]("nd_address1")
        val ndAddress2 = cb.newLocal[Long]("nd_address2")
        val r2PointerToNDAddress1 = cb.newLocal[Long]("r2_ptr_to_nd_addr1")

        val shapeSeq = IndexedSeq(const(3L))
        val shapeMaker = nd.makeShapeBuilder(shapeSeq)
        val shapeSRVB = new StagedRegionValueBuilder(cb.emb, nd.shape.pType, codeRegion1)
        cb.append(shapeMaker(shapeSRVB))
        cb.assign(shapeAddress, shapeSRVB.end())
        val shapePCode = PCode(nd.shape.pType, shapeAddress).asBaseStruct
        val shapePValue = shapePCode.memoize(cb, "testRefCounted_shape")

        // Region 1 just gets 1 ndarray.
        cb.assign(ndAddress1, nd.allocateAndInitialize(cb, codeRegion1, shapePValue, shapePValue))

        // Region 2 gets an ndarray at ndaddress2, plus a reference to the one at ndarray 1.
        cb.assign(ndAddress2, nd.allocateAndInitialize(cb, codeRegion2, shapePValue, shapePValue))
        cb.assign(r2PointerToNDAddress1, codeRegion2.allocate(8L, 8L))
        cb.append(nd.constructAtAddress(cb.emb, r2PointerToNDAddress1, codeRegion2, nd, ndAddress1, true))
        ndAddress1
      }
    } catch {
      case e: AssertionError =>
        region1.clear()
        throw e
    }

    val f = fb.result()()
    val result1 = f(region1, region2, region3)

    // Check number of ndarrays in each region:
    assert(region1.memory.listNDArrayRefs().size == 1)
    assert(region1.memory.listNDArrayRefs()(0) == result1)

    assert(region2.memory.listNDArrayRefs().size == 2)
    assert(region2.memory.listNDArrayRefs()(1) == result1)

    // Check that the reference count of ndarray1 is 2:
    val rc1A = Region.loadLong(result1-16L)
    assert(rc1A == 2)

    region1.clear()
    assert(region1.memory.listNDArrayRefs().size == 0)

    // Check that ndarray 1 wasn't actually cleared, ref count should just be 1 now:
    val rc1B = Region.loadLong(result1-16L)
    assert(rc1B == 1)
  }
}
