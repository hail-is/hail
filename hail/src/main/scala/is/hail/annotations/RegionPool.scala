package is.hail.annotations

import is.hail.expr.ir.LongArrayBuilder
import is.hail.utils._

object RegionPool {

  def apply(strictMemoryCheck: Boolean = false): RegionPool = {
    val thread = Thread.currentThread()
    new RegionPool(strictMemoryCheck, thread.getName, thread.getId)
  }

  def scoped[T](f: RegionPool => T): T = using(RegionPool(false))(f)
}

final class RegionPool private(strictMemoryCheck: Boolean, threadName: String, threadID: Long) extends AutoCloseable {
  log.info(s"RegionPool: initialized for thread $threadID: $threadName")
  protected[annotations] val freeBlocks: Array[LongArrayBuilder] = Array.fill[LongArrayBuilder](4)(new LongArrayBuilder(8))
  protected[annotations] val regions = new BoxedArrayBuilder[RegionMemory]()
  private[this] val freeRegions = new BoxedArrayBuilder[RegionMemory]()
  private[this] val blocks: Array[Long] = Array(0L, 0L, 0L, 0L)
  private[this] var totalAllocatedBytes: Long = 0L
  private[this] var allocationEchoThreshold: Long = 256 * 1024
  private[this] var numJavaObjects: Long = 0L
  private[this] var maxNumJavaObjects: Long = 0L

  def addJavaObject(): Unit = {
    numJavaObjects += 1
  }

  def removeJavaObjects(n: Int): Unit = {
    numJavaObjects -= n
  }

  def getTotalAllocatedBytes: Long = totalAllocatedBytes

  private[annotations] def incrementAllocatedBytes(toAdd: Long): Unit = {
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

  protected[annotations] def freeChunks(ab: LongArrayBuilder, totalSize: Long): Unit = {
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
    val nFree = this.numFreeRegions()
    val nRegions = this.numRegions()
    val nBlocks = this.numFreeBlocks()

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
      s"${readableBytes(totalAllocatedBytes - inBlocks)} chunks), regions.size = ${regions.size}, " +
      s"$numJavaObjects current java objects, $maxNumJavaObjects max java objects, thread $threadID: $threadName")
//    log.info("-----------STACK_TRACES---------")
//    val stacks: String = regions.result().toIndexedSeq.flatMap(r => r.stackTrace.map((r.getTotalChunkMemory(), _))).foldLeft("")((a: String, b) => a + "\n" + b.toString())
//    log.info(stacks)
//    log.info("---------------END--------------")


  }

  def scopedRegion[T](f: Region => T): T = using(Region(pool = this))(f)
  def scopedSmallRegion[T](f: Region => T): T = using(Region(Region.SMALL, pool=this))(f)
  def scopedTinyRegion[T](f: Region => T): T = using(Region(Region.TINY, pool=this))(f)

  override def finalize(): Unit = close()

  private[this] var closed: Boolean = false

  def close(): Unit = {
    if (closed)
      return
    closed = true

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
