package is.hail.annotations

import is.hail.TestUtils._
import is.hail.collection.LongArrayBuilder
import is.hail.utils.using

import scala.collection.mutable.ArrayBuffer

import org.junit.jupiter.api.Test

class RegionSuite {

  @Test def testRegionSizes(): Unit =
    RegionPool.scoped { pool =>
      pool.scopedSmallRegion(region => Array.range(0, 30).foreach(_ => region.allocate(1, 500)))

      pool.scopedTinyRegion(region => Array.range(0, 30).foreach(_ => region.allocate(1, 60)))
    }

  @Test def testRegionAllocationSimple(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      assertEq(pool.numFreeBlocks(), 0)
      assertEq(pool.numRegions(), 0)
      assertEq(pool.numFreeRegions(), 0)

      val r = pool.getRegion(Region.REGULAR)

      assertEq(pool.numRegions(), 1)
      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 0)

      r.clear()

      assertEq(pool.numRegions(), 1)
      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 0)

      r.allocate(Region.SIZES(Region.REGULAR) - 1): Unit
      r.allocate(16): Unit
      r.clear()

      assertEq(pool.numRegions(), 1)
      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 1)

      val r2 = pool.getRegion(Region.SMALL)

      assertEq(pool.numRegions(), 2)
      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 1)

      val r3 = pool.getRegion(Region.REGULAR)

      assertEq(pool.numRegions(), 3)
      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 0)

      r.invalidate()
      r2.invalidate()
      r3.invalidate()

      assertEq(pool.numRegions(), 3)
      assertEq(pool.numFreeRegions(), 3)
      assertEq(pool.numFreeBlocks(), 3)

      val r4 = pool.getRegion(Region.TINIER)

      assertEq(pool.numRegions(), 3)
      assertEq(pool.numFreeRegions(), 2)
      assertEq(pool.numFreeBlocks(), 3)

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
        assertEq(after, c)
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

      using(region.getParentReference(2, Region.TINY))(r => assertEq(offset(r), off2))

      using(region.getParentReference(4, Region.SMALL))(r => assertEq(offset(r), off4))

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
      region.invalidate()
      assertEq(off2 - off1, 10L)
    }
  }

  @Test def blocksAreNotReleasedUntilRegionIsReleased(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      val nBlocks = 5
      (0 until (Region.SIZES(Region.REGULAR)).toInt * nBlocks by 256).foreach { _ =>
        region.allocate(1, 256)
      }
      assertEq(pool.numFreeBlocks(), 0)
      region.invalidate()
      assertEq(pool.numFreeBlocks(), 5)
    }
  }

  @Test def largeChunksAreNotReturnedToBlockPool(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      region.allocate(4, Region.SIZES(Region.REGULAR) - 4): Unit

      assertEq(pool.numFreeBlocks(), 0)
      region.allocate(4, 1024 * 1024): Unit
      region.invalidate()
      assertEq(pool.numFreeBlocks(), 1)
    }
  }

  @Test def referencedRegionsAreNotFreedUntilReferencingRegionIsFreed(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val r1 = pool.getRegion()
      val r2 = pool.getRegion()
      r2.addReferenceTo(r1)
      r1.invalidate()
      assertEq(pool.numRegions(), 2)
      assertEq(pool.numFreeRegions(), 0)
      r2.invalidate()
      assertEq(pool.numRegions(), 2)
      assertEq(pool.numFreeRegions(), 2)
    }
  }

  @Test def blockSizesWorkAsExpected(): Unit = {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 0)

      val region1 = pool.getRegion()
      assertEq(region1.blockSize, Region.REGULAR)
      region1.invalidate()

      assertEq(pool.numFreeRegions(), 1)
      assertEq(pool.numFreeBlocks(), 1)

      val region2 = pool.getRegion(Region.SMALL)
      assertEq(region2.blockSize, Region.SMALL)

      assertEq(pool.numFreeRegions(), 0)
      assertEq(pool.numFreeBlocks(), 1)

      region2.invalidate()

      assertEq(pool.numFreeRegions(), 1)
      assertEq(pool.numFreeBlocks(), 2)
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
      assertEq(operations(0), ("allocate", 512L))
      // 512 size chunk freed from cache to not exceed peak memory
      assertEq(operations(1), ("free", 0L))
      assertEq(operations(2), ("allocate", 64L))
      chunkCache.freeChunkToCache(ab.pop())
      // No additional allocate should be made as uses cache
      ab += chunkCache.getChunk(pool, 50L)._1
      assertEq(operations.length, 3)
      ab += chunkCache.getChunk(pool, 40L)._1
      chunkCache.freeChunksToCache(ab)
      assertEq(operations(3), ("allocate", 64L))
      assertEq(operations.length, 4)
      chunkCache.freeAll(pool)
      assertEq(operations(4), ("free", 0L))
      assertEq(operations(5), ("free", 0L))
      assertEq(operations.length, 6)
    }
  }
}
