package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, SafeNDArray, ScalaToRegionValue, UnsafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.utils._
import org.apache.spark.sql.Row
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
        Region.loadAddress(r2PointerToNDAddress1)
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
    val unsafe1 = UnsafeRow.read(pNd, region1, addr1)
    val unsafe2 = UnsafeRow.read(pNd, region2, addr2)
    // Deep copy same ptype just increments reference count, doesn't change the address.
    assert(addr1 == addr2)
    assert(PNDArray.getReferenceCount(addr1) == 2)
    assert(unsafe1 == unsafe2)
    region1.clear()
    assert(PNDArray.getReferenceCount(addr1) == 1)

    // Deep copy with elements that contain pointers, so have to actually do a full copy
    val pNDOfArrays = PCanonicalNDArray(PCanonicalArray(PInt32Required, true), 1)
    val annotationNDOfArrays = new SafeNDArray(IndexedSeq(3L), (0 until 3).map(idx => (0 to idx).toArray.toIndexedSeq))
    val addr3 = pNDOfArrays.unstagedStoreJavaObject(annotationNDOfArrays, region=region1)
    val unsafe3 = UnsafeRow.read(pNDOfArrays, region1, addr3)
    val addr4 = pNDOfArrays.copyFromAddress(region2, pNDOfArrays, addr3, true)
    val unsafe4 = UnsafeRow.read(pNDOfArrays, region2, addr4)
    assert(addr3 != addr4)
    assert(unsafe3 == unsafe4)
    assert(PNDArray.getReferenceCount(addr3) == 1L)
    assert(PNDArray.getReferenceCount(addr4) == 1L)

    // Deep copy with PTypes with different requirements
    val pNDOfStructs1 = PCanonicalNDArray(PCanonicalStruct(true, ("x", PInt32Required), ("y", PInt32())), 1)
    val pNDOfStructs2 = PCanonicalNDArray(PCanonicalStruct(true, ("x", PInt32()), ("y", PInt32Required)), 1)
    val annotationNDOfStructs = new SafeNDArray(IndexedSeq(5L), (0 until 5).map(idx => Row(idx, idx + 100)))

    val addr5 = pNDOfStructs1.unstagedStoreJavaObject(annotationNDOfStructs, region=region1)
    val unsafe5 = UnsafeRow.read(pNDOfStructs1, region1, addr5)
    val addr6 = pNDOfStructs2.copyFromAddress(region2, pNDOfStructs1, addr5, true)
    val unsafe6 = UnsafeRow.read(pNDOfStructs2, region2, addr6)
    assert(unsafe5 == unsafe6)
  }
}
