package is.hail.annotations

import is.hail.utils._
import java.lang.ref.{ PhantomReference, ReferenceQueue }
import scala.collection.mutable

object RegionPool {
  private[this] lazy val thePool: ThreadLocal[RegionPool] = new ThreadLocal[RegionPool]() {
    override def initialValue(): RegionPool = RegionPool()
  }

  def get: RegionPool = thePool.get()

  def apply(strictMemoryCheck: Boolean = false): RegionPool = {
    val thread = Thread.currentThread()
    val pool = new RegionPool(
      new RegionPoolNativeMemoryOwner(
        strictMemoryCheck, thread.getName, thread.getId))
    RegionPoolNativeMemoryFreer.track(pool)
    pool
  }
}

object RegionPoolNativeMemoryFreer {
  private[this] var cleaner: Thread = null
  private[this] var queue: ReferenceQueue[RegionPool] = null
  private[this] var refs: mutable.Set[RegionPoolNativeMemoryFreer] = null

  def track(pool: RegionPool) = this.synchronized {
    if (cleaner == null) {
      queue = new ReferenceQueue()
      refs = mutable.Set()
      cleaner = new Thread(new Runnable() {
        def run() {
          while (true) {
            val phantomRef = queue.remove().asInstanceOf[RegionPoolNativeMemoryFreer]
            phantomRef.nativeMemoryOwner.free()
            refs.remove(phantomRef)
          }
        }
      })
      cleaner.start()
    }

    refs.add(new RegionPoolNativeMemoryFreer(pool, queue))
  }
}

final class RegionPoolNativeMemoryFreer (
  pool: RegionPool,
  queue: ReferenceQueue[RegionPool]
) extends PhantomReference[RegionPool](pool, queue) {
  private val nativeMemoryOwner = pool.referent
}

final class RegionPool private (
  protected[annotations] val referent: RegionPoolNativeMemoryOwner
) {
  def getTotalAllocatedBytes: Long = referent.getTotalAllocatedBytes
  protected[annotations] def reclaim(memory: RegionMemory): Unit = referent.reclaim(memory)
  protected[annotations] def getBlock(size: Int): Long = referent.getBlock(size)
  protected[annotations] def getChunk(size: Long): Long = referent.getChunk(size)
  protected[annotations] def freeChunks(ab: ArrayBuilder[Long], totalSize: Long): Unit = referent.freeChunks(ab, totalSize)
  protected[annotations] def getMemory(size: Int): RegionMemory = referent.getMemory(size, this)
  val freeBlocks = referent.freeBlocks
  def getRegion(): Region = referent.getRegion(this)
  def getRegion(size: Int): Region = referent.getRegion(size, this)
  def numRegions(): Int = referent.numRegions()
  def numFreeRegions(): Int = referent.numFreeRegions()
  def numFreeBlocks(): Int = referent.numFreeBlocks()
  def logStats(context: String): Unit = referent.logStats(context)
}

final class RegionPoolNativeMemoryOwner protected[annotations] (
  strictMemoryCheck: Boolean,
  threadName: String,
  threadID: Long
) {
  log.info(s"RegionPool: initialized for thread $threadID: $threadName")
  protected[annotations] val freeBlocks: Array[ArrayBuilder[Long]] = Array.fill[ArrayBuilder[Long]](4)(new ArrayBuilder[Long])
  protected[annotations] val regions = new ArrayBuilder[RegionMemory]()
  private[this] val freeRegions = new ArrayBuilder[RegionMemory]()
  private[this] val blocks: Array[Long] = Array(0L, 0L, 0L, 0L)
  private[this] var totalAllocatedBytes: Long = 0L
  private[this] var allocationEchoThreshold: Long = 256 * 1024

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

  protected[annotations] def getMemory(size: Int, pool: RegionPool): RegionMemory = {
    if (freeRegions.size > 0) {
      val rm = freeRegions.pop()
      rm.initialize(size)
      rm
    } else {
      val rm = new RegionMemory(pool)
      rm.initialize(size)
      regions += rm
      rm
    }
  }

  def getRegion(pool: RegionPool): Region = getRegion(Region.REGULAR, pool)

  def getRegion(size: Int, pool: RegionPool): Region = {
    val r = new Region(size, pool)
    r.memory = getMemory(size, pool)
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

  private[this] def report(context: String): Unit = {
    var inBlocks = 0L
    var i = 0
    while (i < 4) {
      inBlocks += blocks(i) * Region.SIZES(i)
      i += 1
    }

    log.info(s"RegionPool: $context: ${readableBytes(totalAllocatedBytes)} allocated (${readableBytes(inBlocks)} blocks / " +
      s"${readableBytes(totalAllocatedBytes - inBlocks)} chunks), thread $threadID: $threadName")
  }

  def free(): Unit = {
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
