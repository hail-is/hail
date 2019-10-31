package is.hail.annotations

import is.hail.utils._

object RegionPool {
  private lazy val thePool: ThreadLocal[RegionPool] = new ThreadLocal[RegionPool]() {
    override def initialValue(): RegionPool = RegionPool()
  }

  def get: RegionPool = thePool.get()

  def apply(strictMemoryCheck: Boolean = false): RegionPool = {
    val thread = Thread.currentThread()
    new RegionPool(strictMemoryCheck, thread.getName, thread.getId)
  }
}

final class RegionPool private(strictMemoryCheck: Boolean, threadName: String, threadID: Long) extends AutoCloseable {
  log.info(s"RegionPool: initialized for thread $threadID: $threadName")
  protected[annotations] val freeBlocks: Array[ArrayBuilder[Long]] = Array.fill[ArrayBuilder[Long]](4)(new ArrayBuilder[Long])
  protected[annotations] val regions = new ArrayBuilder[RegionMemory]()
  private val freeRegions = new ArrayBuilder[RegionMemory]()
  private val blocks: Array[Long] = Array(0L, 0L, 0L, 0L)
  private var totalAllocatedBytes: Long = 0L
  private var allocationEchoThreshold: Long = 256 * 1024

  def getTotalAllocatedBytes: Long = totalAllocatedBytes

  private def incrementAllocatedBytes(toAdd: Long): Unit = {
    totalAllocatedBytes += toAdd
    if (totalAllocatedBytes >= allocationEchoThreshold) {
      report("REPORT_THRESHOLD")
      allocationEchoThreshold *= 2
    }
  }

  protected[annotations] def reclaim(memory: RegionMemory): Unit = {
    freeRegions += memory
  }

  protected[annotations] def getBlock(size: Int): Long = {
    val pool = freeBlocks(size)
    if (pool.size > 0) {
      pool.pop()
    } else {
      blocks(size) += 1
      val blockByteSize = Region.SIZES(size)
      incrementAllocatedBytes(blockByteSize)
      Memory.malloc(blockByteSize)
    }
  }

  protected[annotations] def getChunk(size: Long): Long = {
    incrementAllocatedBytes(size)
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
    val usedBlockCounts = blocks.zip(freeBlockCounts).map { case (tot, free) => tot - free }
    info(
      s"""Region count for $context
         |    regions: $nRegions active, $nFree free
         |     blocks: $nBlocks
         |       free: ${ freeBlockCounts.mkString(", ") }
         |       used: ${ usedBlockCounts.mkString(", ") }""".stripMargin)
  }

  def report(context: String): Unit = {
    var inBlocks = 0L
    var i = 0
    while (i < 4) {
      inBlocks += blocks(i) * Region.SIZES(i)
      i += 1
    }

    log.info(s"RegionPool: $context: ${readableBytes(totalAllocatedBytes)} allocated (${readableBytes(inBlocks)} blocks / " +
      s"${readableBytes(totalAllocatedBytes - inBlocks)} chunks), thread $threadID: $threadName")
  }

  override def finalize(): Unit = close()

  def close(): Unit = {
    report("FREE")

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