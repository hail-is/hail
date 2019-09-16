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

final class RegionPool(strictMemoryCheck: Boolean = false) extends AutoCloseable {
  protected[annotations] val freeBlocks: Array[ArrayBuilder[Long]] = Array.fill[ArrayBuilder[Long]](4)(new ArrayBuilder[Long])
  protected[annotations] val regions = new ArrayBuilder[RegionMemory]()
  private val freeRegions = new ArrayBuilder[RegionMemory]()

  private val nBlocksAllocated: Array[Long] = Array(0L, 0L, 0L, 0L)

  private var totalAllocatedBytes: Long = 0L

  def getTotalAllocatedBytes: Long = totalAllocatedBytes

  protected[annotations] def reclaim(memory: RegionMemory): Unit = {
    freeRegions += memory
  }

  protected[annotations] def getBlock(size: Int): Long = {
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

  protected[annotations] def getChunk(size: Long): Long = {
    totalAllocatedBytes += size
    Memory.malloc(size)
  }

  protected[annotations] def freeChunks(ab: ArrayBuilder[Long], totalSize: Long): Unit = {
    while (ab.size > 0) {
      val addr = ab.pop()
      Memory.free(addr)
    }
    totalAllocatedBytes -= totalSize
  }

  protected[annotations] def getMemory(size: Int): RegionMemory = {
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

  def getRegion(): Region = getRegion(Region.REGULAR)

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
    log.info(s"freeing RegionPool with $totalAllocatedBytes bytes allocated")

    var i = 0
    while (i < regions.size) {
      regions(i).freeMemory()
      i += 1
    }

    i = 0
    while (i < 4) {
      val blockSize = Region.SIZES(i)
      val blocks = freeBlocks(i)
      while (blocks.size > 0) {
        Memory.free(blocks.pop())
        totalAllocatedBytes -= blockSize
      }
      i += 1
    }

    if (totalAllocatedBytes != 0) {
      val msg = s"RegionPool: total allocated bytes not 0 after closing! total allocated: " +
        s"$totalAllocatedBytes (${ readableBytes(totalAllocatedBytes) })"
      if (strictMemoryCheck)
        fatal(msg)
      else
        warn(msg)
    }
  }
}