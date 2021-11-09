package is.hail.annotations

import is.hail.expr.ir.LongArrayBuilder

import java.util.TreeMap
import java.util.function.BiConsumer
import scala.collection.mutable

  /**
    *     ChunkCache minimizes calls to free and allocate by holding onto
    *  chunks when they are no longer in use. When a chunk is needed, the cache
    *  is searched. If the size requested is less than a certain amount, the size
    *  is rounded up to the nearest power of 2 and the small chunk cache is checked
    *  for available chunk. If bigger, the big chunk cache returns the chunk whose size
    *  is the ceiling match. If the size requested is at least 90 percent of the size of
    *  the chunk returned, then that chunk is used. If no acceptable chunk is found, a new
    *  chunk is created. If the chunk created plus the current allocation is greater than
    *  peak usage, than chunks from the cache are deallocated until this condition is not
    *  true or the cache is empty.
    *     When freeChunk is called on RegionPool, the chunks get put in the cache that
    *  corresponds to their size. freeAll releases all chunks and is called when
    *  RegionPool is closed.
    */

private class ChunkCache (allocator: Long => Long, freer: Long => Unit){
  private[this] val highestSmallChunkPowerOf2 = 24
  private[this] val biggestSmallChunk = Math.pow(2,highestSmallChunkPowerOf2)
  private[this] val bigChunkCache = new TreeMap[Long, LongArrayBuilder]()
  private[this] val chunksEncountered = mutable.Map[Long, Long]()
  private[this] val minSpaceRequirements = .9
  private[this] var chunksRequested = 0
  private[this] var cacheHits = 0
  private[this] var smallChunkCacheSize = 0
  private[this] val smallChunkCache = new Array[LongArrayBuilder](highestSmallChunkPowerOf2 + 1)
  (0 until highestSmallChunkPowerOf2 + 1).foreach(index => {
    smallChunkCache(index) = new LongArrayBuilder()
  })

  def getChunkSize(chunkPointer: Long): Long =  chunksEncountered(chunkPointer)

  def freeChunkFromMemory(pool: RegionPool, chunkPointer: Long):Unit = {
    val size = chunksEncountered(chunkPointer)
    pool.decrementAllocatedBytes(size)
    freer(chunkPointer)
    chunksEncountered -= chunkPointer
  }

  def freeChunksFromCacheToFit(pool: RegionPool, sizeToFit: Long): Unit = {
    var smallChunkIndex = highestSmallChunkPowerOf2
    while((sizeToFit + pool.getTotalAllocatedBytes) > pool.getHighestTotalUsage &&
      smallChunkIndex >= 0 && !chunksEncountered.isEmpty) {
      if (!bigChunkCache.isEmpty) {
        val toFree = bigChunkCache.lastEntry()
        freeChunkFromMemory(pool, toFree.getValue.pop())
        if (toFree.getValue.size == 0) bigChunkCache.remove(toFree.getKey)
      }
      else {
        if (smallChunkCacheSize == 0) smallChunkIndex = -1
        else {
          val toFree = smallChunkCache(smallChunkIndex)
          if (toFree.size != 0) {
            freeChunkFromMemory(pool, toFree.pop())
            smallChunkCacheSize -= 1
          }
          if (toFree.size == 0) smallChunkIndex -= 1
        }
      }
    }
  }

  def newChunk(pool: RegionPool, size: Long): Long = {
    if ((size + pool.getTotalAllocatedBytes) > pool.getHighestTotalUsage) {
      freeChunksFromCacheToFit(pool, size)
    }
    pool.incrementAllocatedBytes(size)
    val newChunkPointer = allocator(size)
    chunksEncountered += (newChunkPointer -> size)
    newChunkPointer
  }

  def freeAll(pool: RegionPool): Unit = {
    if (!chunksEncountered.isEmpty) {
      smallChunkCache.foreach(ab => {
        while (ab.size > 0) {
          freeChunkFromMemory(pool, ab.pop())
          smallChunkCacheSize -= 1
        }})
      //BiConsumer needed to work with scala 2.11.12
      bigChunkCache.forEach(new BiConsumer[Long, LongArrayBuilder]() {
        def accept(key: Long, value: LongArrayBuilder): Unit =
          while (value.size > 0) freeChunkFromMemory(pool, value.pop())
      })
    }
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

  def getChunk(pool: RegionPool, size: Long): (Long, Long) = {
    chunksRequested += 1
    assert(size > 0L)
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
        || ((closestSize.getKey * minSpaceRequirements) <= size))) {
        cacheHits += 1
        val chunkPointer = closestSize.getValue.pop()
        if (closestSize.getValue.size == 0) bigChunkCache.remove(closestSize.getKey)
        (chunkPointer, size)
      }
      else (newChunk(pool, size), size)
    }
  }

  def freeChunkToCache(chunkPointer: Long): Unit = {
    val chunkSize = chunksEncountered(chunkPointer)
    if (chunkSize <= biggestSmallChunk) {
      smallChunkCache(indexInSmallChunkCache(chunkSize)) += chunkPointer
      smallChunkCacheSize += 1
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

  def freeChunksToCache( ab: LongArrayBuilder): Unit = {
    while (ab.size > 0) freeChunkToCache(ab.pop())
  }
}


