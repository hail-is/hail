package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, SafeNDArray, ScalaToRegionValue, UnsafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.utils._
import org.testng.annotations.Test

class PNDArraySuite extends PhysicalTestUtils {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      copyTestExecutor(PCanonicalNDArray(PInt64(true), 1), PCanonicalNDArray(PInt64(true), 1), new SafeNDArray(IndexedSeq(3L), IndexedSeq(4L,5L,6L)),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }

  @Test def testRefCounted(): Unit = {
    val nd = PCanonicalNDArray(PInt32Required, 1)

    val region1 = Region(pool=this.pool)
    val region2 = Region(pool=this.pool)
    val region3 = Region(pool=this.pool)
    val fb = EmitFunctionBuilder[Region, Region, Region, Long](ctx, "ref_count_test")
    val codeRegion1 = fb.getCodeParam[Region](1)
    val codeRegion2 = fb.getCodeParam[Region](2)
    val codeRegion3 = fb.getCodeParam[Region](3)

    try {
      fb.emitWithBuilder{ cb =>
        val r2PointerToNDAddress1 = cb.newLocal[Long]("r2_ptr_to_nd_addr1")

        val shapeSeq = IndexedSeq(const(3L))

        // Region 1 just gets 1 ndarray.
        val (_, snd1Finisher) = nd.constructDataFunction(shapeSeq, shapeSeq, cb, codeRegion1)

        val snd1 = snd1Finisher(cb).memoize(cb, "snd1")

        // Region 2 gets an ndarray at ndaddress2, plus a reference to the one at ndarray 1.
        val (_, snd2Finisher) = nd.constructDataFunction(shapeSeq, shapeSeq, cb, codeRegion2)
        val snd2 = snd2Finisher(cb).memoize(cb, "snd2")
        cb.assign(r2PointerToNDAddress1, codeRegion2.allocate(8L, 8L))

        nd.storeAtAddress(cb, r2PointerToNDAddress1, codeRegion2, snd1, true)

        // Return the address of the 1st one
        snd1.tcode[Long]
      }
    } catch {
      case e: AssertionError =>
        region1.clear()
        region2.clear()
        region3.clear()
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


    assert(region3.memory.listNDArrayRefs().size == 0)
    // Do an unstaged copy into region3
    nd.copyFromAddress(region3, nd, result1, true)
    assert(region3.memory.listNDArrayRefs().size == 1)

    // Check that clearing region2 removes both ndarrays
    region2.clear()
    assert(region2.memory.listNDArrayRefs().size == 0)
  }

  @Test def testUnstagedCopy(): Unit = {
    val region1 = Region(pool=this.pool)
    val region2 = Region(pool=this.pool)
    val x = new SafeNDArray(IndexedSeq(3L, 2L), (0 until 6).map(_.toDouble))
    val pNd = PCanonicalNDArray(PFloat64Required, 2, true)
    val addr1 = pNd.unstagedStoreJavaObject(x, region=region1)
    val addr2 = pNd.copyFromAddress(region2, pNd, addr1, true)
    // Deep copy just increments reference count, doesn't change the address.
    assert(addr1 == addr2)
    assert(PNDArray.getReferenceCount(addr1) == 2)
  }
}
