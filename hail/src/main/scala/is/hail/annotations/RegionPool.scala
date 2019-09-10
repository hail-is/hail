package is.hail.annotations

import is.hail.utils._

object RegionPool {
  private val pools = new java.util.concurrent.ConcurrentHashMap[Long, RegionPool]()

  def get: RegionPool = {
    val makePool: java.util.function.Function[Long, RegionPool] = new java.util.function.Function[Long, RegionPool] {
      def apply(id: Long): RegionPool = new RegionPool()
    }
    pools.computeIfAbsent(Thread.currentThread().getId(), makePool)
  }
}

final class RegionPool extends AutoCloseable {
  val freeBlocks: Array[ArrayBuilder[Long]] = Array.fill[ArrayBuilder[Long]](4)(new ArrayBuilder[Long])
  val regions = new ArrayBuilder[RegionMemory]()
  private val freeRegions = new ArrayBuilder[RegionMemory]()

  private val nBlocksAllocated: Array[Long] = Array(0L, 0L, 0L, 0L)

  private var totalAllocatedBytes: Long = 0L

  def getTotalAllocatedBytes: Long = totalAllocatedBytes

  def reclaim(memory: RegionMemory): Unit = {
    freeRegions += memory
  }

  private var id = 0L

  def nextID: Long = {
    id += 1
    id
  }

  def getBlock(size: Int): Long = {
    val pool = freeBlocks(size)
    if (pool.size > 0) {
      pool.pop()
    } else {
      nBlocksAllocated(size) += 1
      val blockByteSize = Region.SIZES(size)
      totalAllocatedBytes += blockByteSize
      Memory.malloc(blockByteSize)
    }
  }

  def getChunk(size: Long): Long = {
    totalAllocatedBytes += size
    Memory.malloc(size)
  }

  def freeChunk(addr: Long): Unit = {
    totalAllocatedBytes -= addr
    Memory.free(addr)
  }

  def getBlock(): Long = getBlock(0)

  def getMemory(size: Int): RegionMemory = {
    if (freeRegions.size > 0) {
      val rm = freeRegions.pop()
      rm.initialize(size)
      rm
    } else {
      val rm = new RegionMemory(this)
      rm.initialize(size)
      regions += rm
      rm
    }
  }

  def getRegion(size: Int): Region = {
    val r = new Region(size, this)
    r.memory = getMemory(size)
    r
  }

  def numRegions(): Int = regions.size

  def numFreeRegions(): Int = freeRegions.size

  def numFreeBlocks(): Int = freeBlocks.map(_.size).sum

  def logStats(context: String): Unit = {
    val pool = RegionPool.get
    val nFree = pool.numFreeRegions()
    val nRegions = pool.numRegions()
    val nBlocks = pool.numFreeBlocks()

    val freeBlockCounts = freeBlocks.map(_.size)
    val usedBlockCounts = nBlocksAllocated.zip(freeBlockCounts).map { case (tot, free) => tot - free }
    info(
      s"""Region count for $context
         |    regions: $nRegions active, $nFree free
         |     blocks: $nBlocks
         |       free: ${ freeBlockCounts.mkString(", ") }
         |       used: ${ usedBlockCounts.mkString(", ") }""".stripMargin)
  }

  def close(): Unit = {
    info(s"freeing RegionPool with $totalAllocatedBytes bytes allocated")

    var i = 0
    while (i < regions.size) {
      regions(i).freeMemory()
      i += 1
    }

    i = 0
    while (i < 4) {
      val blockSize = Region.SIZES(i)
      info(s"at blockSize ${ blockSize }")
      val blocks = freeBlocks(i)
      while (blocks.size > 0) {
        Memory.free(blocks.pop())
        totalAllocatedBytes -= blockSize
        info(s"freed ${ blockSize } bytes to $totalAllocatedBytes")
      }
      i += 1
    }

    assert(totalAllocatedBytes == 0, totalAllocatedBytes)
  }
}