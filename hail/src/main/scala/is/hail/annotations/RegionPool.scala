package is.hail.annotations

import is.hail.expr.ir.LongArrayBuilder
import is.hail.utils._

import java.util.TreeMap
import java.util.function.BiConsumer
import scala.collection.mutable

object RegionPool {

  def apply(strictMemoryCheck: Boolean = false): RegionPool = {
    val thread = Thread.currentThread()
    new RegionPool(strictMemoryCheck, thread.getName, thread.getId)
  }

  def scoped[T](f: RegionPool => T): T = using(RegionPool(false))(f)

  lazy val maxRegionPoolSize: Long = {
    val s = System.getenv("HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB")
    if (s != null && s.nonEmpty)
      s.toLong * 1024 * 1024
    else
      Long.MaxValue
  }
}

final class RegionPool private (strictMemoryCheck: Boolean, threadName: String, threadID: Long)
    extends AutoCloseable {
  log.info(s"RegionPool: initialized for thread $threadID: $threadName")

  protected[annotations] val freeBlocks: Array[LongArrayBuilder] =
    Array.fill[LongArrayBuilder](4)(new LongArrayBuilder(8))

  protected[annotations] val regions = new BoxedArrayBuilder[RegionMemory]()
  private[this] val freeRegions = new BoxedArrayBuilder[RegionMemory]()
  private[this] val blocks: Array[Long] = Array(0L, 0L, 0L, 0L)
  private[this] var totalAllocatedBytes: Long = 0L
  private[this] var allocationEchoThreshold: Long = 256 * 1024
  private[this] var numJavaObjects: Long = 0L
  private[this] var highestTotalUsage = 0L
  private[this] val chunkCache = new ChunkCache(Memory.malloc, Memory.free)
  private[this] val maxSize = RegionPool.maxRegionPoolSize

  def addJavaObject(): Unit =
    numJavaObjects += 1

  def removeJavaObjects(n: Int): Unit =
    numJavaObjects -= n

  def getTotalAllocatedBytes: Long = totalAllocatedBytes

  def getHighestTotalUsage: Long = highestTotalUsage
  def getUsage: (Int, Int) = chunkCache.getUsage()

  private[annotations] def decrementAllocatedBytes(toSubtract: Long): Unit =
    totalAllocatedBytes -= toSubtract

  def closeAndThrow(msg: String): Unit = {
    close()
    fatal(msg)
  }

  private[annotations] def incrementAllocatedBytes(toAdd: Long): Unit = {
    totalAllocatedBytes += toAdd
    if (totalAllocatedBytes >= allocationEchoThreshold) {
      report("REPORT_THRESHOLD")
      allocationEchoThreshold *= 2
    }
    if (totalAllocatedBytes >= highestTotalUsage) {
      highestTotalUsage = totalAllocatedBytes
      if (totalAllocatedBytes > maxSize) {
        val inBlocks = bytesInBlocks()
        closeAndThrow(
          s"Hail off-heap memory exceeded maximum threshold: limit ${formatSpace(maxSize)}, allocated ${formatSpace(totalAllocatedBytes)}\n"
            + s"Report: ${readableBytes(totalAllocatedBytes)} allocated (${readableBytes(inBlocks)} blocks / "
            + s"${readableBytes(totalAllocatedBytes - inBlocks)} chunks), regions.size = ${regions.size}, "
            + s"$numJavaObjects current java objects, thread $threadID: $threadName"
        )
      }
    }
  }

  protected[annotations] def reclaim(memory: RegionMemory): Unit =
    freeRegions += memory

  protected[annotations] def getBlock(size: Int): Long = {
    val pool = freeBlocks(size)
    if (pool.size > 0) {
      pool.pop()
    } else {
      chunkCache.freeChunksFromCacheToFit(this, size.toLong)
      val blockByteSize = Region.SIZES(size)
      incrementAllocatedBytes(blockByteSize)
      blocks(size) += 1
      Memory.malloc(blockByteSize)
    }
  }

  protected[annotations] def getChunk(size: Long): (Long, Long) =
    chunkCache.getChunk(this, size)

  protected[annotations] def freeChunks(ab: LongArrayBuilder, totalSize: Long): Unit =
    chunkCache.freeChunksToCache(ab)

  protected[annotations] def freeChunk(chunkPointer: Long): Unit =
    chunkCache.freeChunkToCache(chunkPointer)

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

  def bytesInBlocks(): Long = Region.SIZES.zip(blocks).map { case (size, block) =>
    size * block
  }.sum[Long]

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
         |       free: ${freeBlockCounts.mkString(", ")}
         |       used: ${usedBlockCounts.mkString(", ")}""".stripMargin
    )
  }

  def report(context: String): Unit = {
    val inBlocks = bytesInBlocks()

    log.info(
      s"RegionPool: $context: ${readableBytes(totalAllocatedBytes)} allocated (${readableBytes(inBlocks)} blocks / " +
        s"${readableBytes(totalAllocatedBytes - inBlocks)} chunks), regions.size = ${regions.size}, " +
        s"$numJavaObjects current java objects, thread $threadID: $threadName"
    )
//    log.info("-----------STACK_TRACES---------")
//    val stacks: String = regions.result().toIndexedSeq.flatMap(r => r.stackTrace.map((r.getTotalChunkMemory(), _))).foldLeft("")((a: String, b) => a + "\n" + b.toString())
//    log.info(stacks)
//    log.info("---------------END--------------")
  }

  def scopedRegion[T](f: Region => T): T = using(Region(pool = this))(f)
  def scopedSmallRegion[T](f: Region => T): T = using(Region(Region.SMALL, pool = this))(f)
  def scopedTinyRegion[T](f: Region => T): T = using(Region(Region.TINY, pool = this))(f)

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
        val popped = blocks.pop()
        Memory.free(popped)
        decrementAllocatedBytes(blockSize)
      }
      i += 1
    }
    chunkCache.freeAll(pool = this)
    if (totalAllocatedBytes != 0) {
      val msg = s"RegionPool: total allocated bytes not 0 after closing! total allocated: " +
        s"$totalAllocatedBytes (${readableBytes(totalAllocatedBytes)})"
      if (strictMemoryCheck)
        fatal(msg)
      else
        warn(msg)
    }
  }
}
