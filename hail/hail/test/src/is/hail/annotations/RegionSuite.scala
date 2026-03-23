package is.hail.annotations

import is.hail.collection.LongArrayBuilder
import is.hail.utils.using

import scala.collection.mutable.ArrayBuffer

class RegionSuite extends munit.FunSuite {

  test("region sizes") {
    RegionPool.scoped { pool =>
      pool.scopedSmallRegion(region => Array.range(0, 30).foreach(_ => region.allocate(1, 500)))

      pool.scopedTinyRegion(region => Array.range(0, 30).foreach(_ => region.allocate(1, 60)))
    }
  }

  test("region allocation simple") {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      assertEquals(pool.numFreeBlocks(), 0)
      assertEquals(pool.numRegions(), 0)
      assertEquals(pool.numFreeRegions(), 0)

      val r = pool.getRegion(Region.REGULAR)

      assertEquals(pool.numRegions(), 1)
      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 0)

      r.clear()

      assertEquals(pool.numRegions(), 1)
      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 0)

      r.allocate(Region.SIZES(Region.REGULAR) - 1): Unit
      r.allocate(16): Unit
      r.clear()

      assertEquals(pool.numRegions(), 1)
      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 1)

      val r2 = pool.getRegion(Region.SMALL)

      assertEquals(pool.numRegions(), 2)
      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 1)

      val r3 = pool.getRegion(Region.REGULAR)

      assertEquals(pool.numRegions(), 3)
      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 0)

      r.invalidate()
      r2.invalidate()
      r3.invalidate()

      assertEquals(pool.numRegions(), 3)
      assertEquals(pool.numFreeRegions(), 3)
      assertEquals(pool.numFreeBlocks(), 3)

      val r4 = pool.getRegion(Region.TINIER)

      assertEquals(pool.numRegions(), 3)
      assertEquals(pool.numFreeRegions(), 2)
      assertEquals(pool.numFreeBlocks(), 3)

      r4.invalidate()
    }
  }

  test("region allocation") {
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
        assertEquals(after, c)
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

  test("region references") {
    RegionPool.scoped { pool =>
      def offset(region: Region) = region.allocate(0)

      def numUsed(): Int = pool.numRegions() - pool.numFreeRegions()

      def assertUsesRegions[T](n: Int)(f: => T): T = {
        val usedRegionCount = numUsed()
        val res = f
        assertEquals(usedRegionCount, numUsed() - n)
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

      using(region.getParentReference(2, Region.TINY))(r => assertEquals(offset(r), off2))

      using(region.getParentReference(4, Region.SMALL))(r => assertEquals(offset(r), off4))

      assertUsesRegions(-1) {
        region.unreferenceRegionAtIndex(2)
      }
      assertUsesRegions(-1) {
        region.unreferenceRegionAtIndex(4)
      }
    }
  }

  test("allocation at start of block is correct") {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      val off1 = region.allocate(1, 10)
      val off2 = region.allocate(1, 10)
      region.invalidate()
      assertEquals(off2 - off1, 10L)
    }
  }

  test("blocks are not released until region is released") {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      val nBlocks = 5
      (0 until (Region.SIZES(Region.REGULAR)).toInt * nBlocks by 256).foreach { _ =>
        region.allocate(1, 256)
      }
      assertEquals(pool.numFreeBlocks(), 0)
      region.invalidate()
      assertEquals(pool.numFreeBlocks(), 5)
    }
  }

  test("large chunks are not returned to block pool") {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val region = pool.getRegion(Region.REGULAR)
      region.allocate(4, Region.SIZES(Region.REGULAR) - 4): Unit

      assertEquals(pool.numFreeBlocks(), 0)
      region.allocate(4, 1024 * 1024): Unit
      region.invalidate()
      assertEquals(pool.numFreeBlocks(), 1)
    }
  }

  test("referenced regions are not freed until referencing region is freed") {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      val r1 = pool.getRegion()
      val r2 = pool.getRegion()
      r2.addReferenceTo(r1)
      r1.invalidate()
      assertEquals(pool.numRegions(), 2)
      assertEquals(pool.numFreeRegions(), 0)
      r2.invalidate()
      assertEquals(pool.numRegions(), 2)
      assertEquals(pool.numFreeRegions(), 2)
    }
  }

  test("block sizes work as expected") {
    using(RegionPool(strictMemoryCheck = true)) { pool =>
      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 0)

      val region1 = pool.getRegion()
      assertEquals(region1.blockSize, Region.REGULAR)
      region1.invalidate()

      assertEquals(pool.numFreeRegions(), 1)
      assertEquals(pool.numFreeBlocks(), 1)

      val region2 = pool.getRegion(Region.SMALL)
      assertEquals(region2.blockSize, Region.SMALL)

      assertEquals(pool.numFreeRegions(), 0)
      assertEquals(pool.numFreeBlocks(), 1)

      region2.invalidate()

      assertEquals(pool.numFreeRegions(), 1)
      assertEquals(pool.numFreeBlocks(), 2)
    }
  }

  test("chunk cache") {
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
      assertEquals(operations(0), (("allocate", 512L)))
      // 512 size chunk freed from cache to not exceed peak memory
      assertEquals(operations(1), (("free", 0L)))
      assertEquals(operations(2), (("allocate", 64L)))
      chunkCache.freeChunkToCache(ab.pop())
      // No additional allocate should be made as uses cache
      ab += chunkCache.getChunk(pool, 50L)._1
      assertEquals(operations.length, 3)
      ab += chunkCache.getChunk(pool, 40L)._1
      chunkCache.freeChunksToCache(ab)
      assertEquals(operations(3), (("allocate", 64L)))
      assertEquals(operations.length, 4)
      chunkCache.freeAll(pool)
      assertEquals(operations(4), (("free", 0L)))
      assertEquals(operations(5), (("free", 0L)))
      assertEquals(operations.length, 6)
    }
  }
}
