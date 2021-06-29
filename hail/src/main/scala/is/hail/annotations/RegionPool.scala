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
}

trait ChunkCache {
  def getChunk(pool:RegionPool, size: Long): (Long, Long)
  def freeChunks(pool: RegionPool, ab: LongArrayBuilder, totalSize: Long): Unit
}

class noCache extends ChunkCache {
  override def getChunk(pool: RegionPool, size: Long): (Long, Long) = {
    pool.incrementAllocatedBytes(size)
    (Memory.malloc(size), size)
  }

  override def freeChunks(pool: RegionPool, ab: LongArrayBuilder, totalSize: Long): Unit = {
    while (ab.size > 0) {
      val addr = ab.pop()
      Memory.free(addr)
    }
    pool.incrementAllocatedBytes(-totalSize)
  }
}

 class ChunkCache1(allocator: Long => Long, freer: Long => Unit) extends ChunkCache {
  val highestSmallChunkPowerOf2 = 24
  val biggestSmallChunk = Math.pow(2,highestSmallChunkPowerOf2)
  val bigChunkCache = new TreeMap[Long, LongArrayBuilder]()
  val chunksEncountered = mutable.Map[Long, Long]()
  val minSpaceRequirements = .9
  var chunksRequested = 0
  var cacheHits = 0
  val smallChunkCache = new Array[LongArrayBuilder](highestSmallChunkPowerOf2 + 1)
   (0 until highestSmallChunkPowerOf2 + 1).foreach(index => {
     smallChunkCache(index) = new LongArrayBuilder()
   })

  def getChunkSize(chunkPointer: Long): Long =  chunksEncountered(chunkPointer)
   def freeInLongArrayBuilder(pool: RegionPool, ab: LongArrayBuilder):Unit = {
     val pointer = ab.pop()
     val size = chunksEncountered(pointer)
     pool.incrementAllocatedBytes(-1 * size)
     freer(pointer)
     chunksEncountered -= pointer
   }
  def deallocateChunksToFit(pool: RegionPool, sizeToFit: Long): Unit = {
    var smallChunkIndex = highestSmallChunkPowerOf2
    while((sizeToFit + pool.getTotalAllocatedBytes) > pool.getHighestTotalUsage &&
          smallChunkIndex >= 0) {
      if (!bigChunkCache.isEmpty) {
        val toFree = bigChunkCache.lastEntry()
        freeInLongArrayBuilder(pool, toFree.getValue)
        if (toFree.getValue.size == 0) bigChunkCache.remove(toFree.getKey)
      }
      else {
        val toFree = smallChunkCache(smallChunkIndex)
        if (toFree.size != 0) {
          freeInLongArrayBuilder(pool, toFree)
        }
        if (toFree.size == 0) smallChunkIndex -= 1
      }
    }
  }
   def newChunk(pool: RegionPool, size: Long): Long = {
     if ((size + pool.getTotalAllocatedBytes) > pool.getHighestTotalUsage) {
       deallocateChunksToFit(pool, size)
     }
     val newChunkPointer = allocator(size)
     chunksEncountered += (newChunkPointer -> size)
     pool.incrementAllocatedBytes(size)
     newChunkPointer
   }
  def freeAll(pool: RegionPool): Unit = {
    smallChunkCache.foreach(ab =>  while(ab.size > 0) freeInLongArrayBuilder(pool, ab))
    //BiConsumer needed to work with scala 2.11.12
    bigChunkCache.forEach( new BiConsumer[Long, LongArrayBuilder]() {
      def accept(key: Long, value: LongArrayBuilder): Unit =
        while(value.size > 0) freeInLongArrayBuilder(pool, value)
    })
  }
  def getUsage(): (Int, Int) = {
    (chunksRequested, cacheHits) 
  }
  def indexInSmallChunkCache(size: Long): Int = {
    var closestPower = highestSmallChunkPowerOf2
    while((size >> closestPower) != 1) closestPower = closestPower - 1
    if (size % (1 << closestPower) != 0) closestPower +=1
    closestPower
  }
  override def getChunk(pool: RegionPool, size: Long): (Long, Long) = {
    chunksRequested += 1
      if (size <= biggestSmallChunk) {
        val closestPower = indexInSmallChunkCache(size)
      if(smallChunkCache(closestPower).size == 0 ) {
        val sizePowerOf2 = (1 << closestPower).toLong
        (newChunk(pool, sizePowerOf2), sizePowerOf2)
      }
      else  {
        cacheHits += 1
        (smallChunkCache(closestPower).pop(), size)
      }
    }
    else {
      val closestSize = bigChunkCache.ceilingEntry(size)
      if (closestSize != null && (closestSize.getKey == size
          || ((closestSize.getKey * .9) <= size))) {
        cacheHits += 1
        val chunkPointer = closestSize.getValue.pop()
        if (closestSize.getValue.size == 0) bigChunkCache.remove(closestSize.getKey)
        (chunkPointer, size)
      }
      else (newChunk(pool, size), size)
    }
  }
  def freeChunk(pool: RegionPool, chunkPointer: Long): Unit = {
    val chunkSize = chunksEncountered(chunkPointer)
    if (chunkSize <= biggestSmallChunk) {
      smallChunkCache(indexInSmallChunkCache(chunkSize)) += chunkPointer
    }
    else {
      val sameSizeEntries = bigChunkCache.get(chunkSize)
      if (sameSizeEntries == null) {
        val newSize = new LongArrayBuilder()
        newSize += chunkPointer
        bigChunkCache.put(chunkSize, newSize)
      }
      else sameSizeEntries += chunkPointer
    }
  }
  override def freeChunks(pool: RegionPool, ab: LongArrayBuilder, totalSize: Long = 0): Unit = {
    while (ab.size > 0) freeChunk(pool, ab.pop())
  }
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
  private[this] var highestTotalUsage = 0L
  private[this] val chunkCache = new ChunkCache1(Memory.malloc, Memory.free)

  def addJavaObject(): Unit = {
    numJavaObjects += 1
  }

  def removeJavaObjects(n: Int): Unit = {
    numJavaObjects -= n
  }

  def getTotalAllocatedBytes: Long = totalAllocatedBytes

  def getHighestTotalUsage: Long = highestTotalUsage
  def getUsage: (Int,Int) = chunkCache.getUsage()
  private[annotations] def incrementAllocatedBytes(toAdd: Long): Unit = {
    totalAllocatedBytes += toAdd
    if (totalAllocatedBytes >= allocationEchoThreshold) {
      report("REPORT_THRESHOLD")
      allocationEchoThreshold *= 2
    }
    if (totalAllocatedBytes >= highestTotalUsage) {
      highestTotalUsage = totalAllocatedBytes
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
      chunkCache.deallocateChunksToFit(this, size.toLong)
      blocks(size) += 1
      val blockByteSize = Region.SIZES(size)
      incrementAllocatedBytes(blockByteSize)
      Memory.malloc(blockByteSize)
    }
  }

  protected[annotations] def getChunk(size: Long): (Long, Long) = {
    chunkCache.getChunk(this, size)
  }

  protected[annotations] def freeChunks(ab: LongArrayBuilder, totalSize: Long): Unit = {
    chunkCache.freeChunks(this, ab, totalSize)
  }
  protected[annotations] def freeChunk(chunkPointer: Long): Unit = {
    chunkCache.freeChunk(this, chunkPointer)
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
      s"$numJavaObjects current java objects, thread $threadID: $threadName")
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
    chunkCache.freeAll(pool = this)
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
