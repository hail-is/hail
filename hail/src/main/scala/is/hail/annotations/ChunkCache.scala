package is.hail.annotations

import is.hail.expr.ir.LongArrayBuilder

import java.util.TreeMap
import java.util.function.BiConsumer
import scala.collection.mutable

class ChunkCache1(allocator: Long => Long, freer: Long => Unit){
  val highestSmallChunkPowerOf2 = 24
  val biggestSmallChunk = Math.pow(2,highestSmallChunkPowerOf2)
  val bigChunkCache = new TreeMap[Long, LongArrayBuilder]()
  val chunksEncountered = mutable.Map[Long, Long]()
  val minSpaceRequirements = .9
  var chunksRequested = 0
  var cacheHits = 0
  var smallChunkCacheSize = 0
  val smallChunkCache = new Array[LongArrayBuilder](highestSmallChunkPowerOf2 + 1)
  (0 until highestSmallChunkPowerOf2 + 1).foreach(index => {
    smallChunkCache(index) = new LongArrayBuilder()
  })

  def getChunkSize(chunkPointer: Long): Long =  chunksEncountered(chunkPointer)

  def freeChunkFromMemory(pool: RegionPool, chunkPointer: Long):Unit = {
    val size = chunksEncountered(chunkPointer)
    pool.incrementAllocatedBytes(-1 * size)
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
    val newChunkPointer = allocator(size)
    chunksEncountered += (newChunkPointer -> size)
    pool.incrementAllocatedBytes(size)
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


