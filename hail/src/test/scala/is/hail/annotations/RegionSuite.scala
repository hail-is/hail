package is.hail.annotations

import is.hail.expr.ir.LongArrayBuilder
import is.hail.utils.using

import scala.collection.mutable.ArrayBuffer

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RegionSuite extends TestNGSuite {

  @Test def testRegionSizes(): Unit =
    RegionPool.scoped { pool =>
      pool.scopedSmallRegion(region => Array.range(0, 30).foreach(_ => region.allocate(1, 500)))

      pool.scopedTinyRegion(region => Array.range(0, 30).foreach(_ => region.allocate(1, 60)))
    }

  @Test def testRegionAllocationSimple(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      assert(pool.numFreeBlocks() == 0)
      assert(pool.numRegions() == 0)
      assert(pool.numFreeRegions() == 0)

      val r = pool.getRegion(Region.REGULAR)

      assert(pool.numRegions() == 1)
      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 0)

      r.clear()

      assert(pool.numRegions() == 1)
      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 0)

      r.allocate(Region.SIZES(Region.REGULAR) - 1)
      r.allocate(16)
      r.clear()

      assert(pool.numRegions() == 1)
      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 1)

      val r2 = pool.getRegion(Region.SMALL)

      assert(pool.numRegions() == 2)
      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 1)

      val r3 = pool.getRegion(Region.REGULAR)

      assert(pool.numRegions() == 3)
      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 0)

      r.invalidate()
      r2.invalidate()
      r3.invalidate()

      assert(pool.numRegions() == 3)
      assert(pool.numFreeRegions() == 3)
      assert(pool.numFreeBlocks() == 3)

      val r4 = pool.getRegion(Region.TINIER)

      assert(pool.numRegions() == 3)
      assert(pool.numFreeRegions() == 2)
      assert(pool.numFreeBlocks() == 3)

      r4.invalidate()
    }
  }

  @Test def testRegionAllocation(): Unit = {
    RegionPool.scoped { pool =>
      case class Counts(regions: Int, freeRegions: Int) {
        def allocate(n: Int): Counts =
          copy(
            regions = regions + math.max(0, n - freeRegions),
            freeRegions = math.max(0, freeRegions - n),
          )

        def free(nRegions: Int, nExtraBlocks: Int = 0): Counts =
          copy(freeRegions = freeRegions + nRegions)
      }

      var before: Counts = null
      var after: Counts = Counts(pool.numRegions(), pool.numFreeRegions())

      def assertAfterEquals(c: => Counts): Unit = {
        before = after
        after = Counts(pool.numRegions(), pool.numFreeRegions())
        assert(after == c)
      }

      pool.scopedRegion { region =>
        assertAfterEquals(before.allocate(1))

        pool.scopedRegion { region2 =>
          assertAfterEquals(before.allocate(1))
          region.addReferenceTo(region2)
        }
        assertAfterEquals(before)
      }
      assertAfterEquals(before.free(2))

      pool.scopedRegion { region =>
        pool.scopedRegion(region2 => region.addReferenceTo(region2))
        pool.scopedRegion(region2 => region.addReferenceTo(region2))
        assertAfterEquals(before.allocate(3))
      }
      assertAfterEquals(before.free(3))
    }
  }

  @Test def testRegionReferences(): Unit = {
    RegionPool.scoped { pool =>
      def offset(region: Region) = region.allocate(0)

      def numUsed(): Int = pool.numRegions() - pool.numFreeRegions()

      def assertUsesRegions[T](n: Int)(f: => T): T = {
        val usedRegionCount = numUsed()
        val res = f
        assert(usedRegionCount == numUsed() - n)
        res
      }

      val region = Region(pool = pool)
      region.setNumParents(5)

      val off4 = using(assertUsesRegions(1) {
        region.getParentReference(4, Region.SMALL)
      })(r => offset(r))

      val off2 = pool.scopedTinyRegion { r =>
        region.setParentReference(r, 2)
        offset(r)
      }

      using(region.getParentReference(2, Region.TINY))(r => assert(offset(r) == off2))

      using(region.getParentReference(4, Region.SMALL))(r => assert(offset(r) == off4))

      assertUsesRegions(-1) {
        region.unreferenceRegionAtIndex(2)
      }
      assertUsesRegions(-1) {
        region.unreferenceRegionAtIndex(4)
      }
    }
  }

  @Test def allocationAtStartOfBlockIsCorrect(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      val off1 = region.allocate(1, 10)
      val off2 = region.allocate(1, 10)
      assert(off2 - off1 == 10)
      region.invalidate()
    }
  }

  @Test def blocksAreNotReleasedUntilRegionIsReleased(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      val nBlocks = 5
      (0 until (Region.SIZES(Region.REGULAR)).toInt * nBlocks by 256).foreach { _ =>
        region.allocate(1, 256)
      }
      assert(pool.numFreeBlocks() == 0)
      region.invalidate()
      assert(pool.numFreeBlocks() == 5)
    }
  }

  @Test def largeChunksAreNotReturnedToBlockPool(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      region.allocate(4, Region.SIZES(Region.REGULAR) - 4)

      assert(pool.numFreeBlocks() == 0)
      region.allocate(4, 1024 * 1024)
      region.invalidate()
      assert(pool.numFreeBlocks() == 1)
    }
  }

  @Test def referencedRegionsAreNotFreedUntilReferencingRegionIsFreed(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val r1 = pool.getRegion()
      val r2 = pool.getRegion()
      r2.addReferenceTo(r1)
      r1.invalidate()
      assert(pool.numRegions() == 2)
      assert(pool.numFreeRegions() == 0)
      r2.invalidate()
      assert(pool.numRegions() == 2)
      assert(pool.numFreeRegions() == 2)
    }
  }

  @Test def blockSizesWorkAsExpected(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 0)

      val region1 = pool.getRegion()
      assert(region1.blockSize == Region.REGULAR)
      region1.invalidate()

      assert(pool.numFreeRegions() == 1)
      assert(pool.numFreeBlocks() == 1)

      val region2 = pool.getRegion(Region.SMALL)
      assert(region2.blockSize == Region.SMALL)

      assert(pool.numFreeRegions() == 0)
      assert(pool.numFreeBlocks() == 1)

      region2.invalidate()

      assert(pool.numFreeRegions() == 1)
      assert(pool.numFreeBlocks() == 2)
    }
  }

  @Test
  def testChunkCache(): Unit = {
    RegionPool.scoped { pool =>
      val operations = ArrayBuffer[(String, Long)]()

      def allocate(numBytes: Long): Long = {
        val pointer = Memory.malloc(numBytes)
        operations += (("allocate", numBytes))
        pointer
      }
      def free(ptrToFree: Long): Unit = {
        operations += (("free", 0L))
        Memory.free(ptrToFree)
      }
      val chunkCache = new ChunkCache(allocate, free)
      val ab = new LongArrayBuilder()

      ab += chunkCache.getChunk(pool, 400L)._1
      chunkCache.freeChunkToCache(ab.pop())
      ab += chunkCache.getChunk(pool, 50L)._1
      assert(operations(0) == (("allocate", 512)))
      // 512 size chunk freed from cache to not exceed peak memory
      assert(operations(1) == (("free", 0L)))
      assert(operations(2) == (("allocate", 64)))
      chunkCache.freeChunkToCache(ab.pop())
      // No additional allocate should be made as uses cache
      ab += chunkCache.getChunk(pool, 50L)._1
      assert(operations.length == 3)
      ab += chunkCache.getChunk(pool, 40L)._1
      chunkCache.freeChunksToCache(ab)
      assert(operations(3) == (("allocate", 64)))
      assert(operations.length == 4)
      chunkCache.freeAll(pool)
      assert(operations(4) == (("free", 0L)))
      assert(operations(5) == (("free", 0L)))
      assert(operations.length == 6)

    }
  }
}
