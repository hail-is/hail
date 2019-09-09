package is.hail.annotations

import is.hail.utils._

final class RegionMemory(pool: RegionPool, var blockSize: Region.Size) extends AutoCloseable {
  private val usedBlocks = new ArrayBuilder[Long](4)
  private val bigChunks = new ArrayBuilder[Long](4)
  private var totalChunkMemory = 0L
  private var currentBlock: Long = pool.getBlock(blockSize)
  private var offsetWithinBlock: Long = 0L
  private val blockThreshold = Math.max(Region.SIZES(blockSize), Region.BLOCK_THRESHOLD)
  private var blockByteSize = Region.SIZES(blockSize)

  private val references = new ArrayBuilder[RegionMemory](4)
  private var referenceCount: Long = 1L

  def allocateNewBlock(): Unit = {
    if (currentBlock != 0)
      usedBlocks += currentBlock
    currentBlock = pool.getBlock(blockSize)
  }

  def getCurrentBlock(): Long = currentBlock

  private def allocateBigChunk(size: Long): Long = {
    val o = pool.getChunk(size)
    bigChunks += o
    totalChunkMemory += size
    // info(s"allocated big chunk of size $size at $o")
    o
  }

  def allocate(n: Long): Long = {
    if (offsetWithinBlock + n <= blockByteSize) {
      val o = currentBlock + offsetWithinBlock
      offsetWithinBlock += n
      o
    } else {
      if (n <= blockThreshold) {
        allocateNewBlock()
        offsetWithinBlock = n
        currentBlock
      } else
        allocateBigChunk(n)
    }
  }

  def allocate(a: Long, n: Long): Long = {
    val alignedOff = (offsetWithinBlock + a - 1) & ~(a - 1);
    val r = if (alignedOff + n <= blockByteSize) {
      val off = currentBlock + alignedOff
      offsetWithinBlock = alignedOff + n
      off
    } else {
      if (n <= blockThreshold) {
        allocateNewBlock()
        offsetWithinBlock = n
        currentBlock
      } else {
        allocateBigChunk(n)
      }
    }
    r
  }

  def freeMemory(): Unit = {
    if (blockSize == -1) {
      assert(references.size == 0)
      assert(usedBlocks.size == 0)
      assert(bigChunks.size == 0)
      return
    }
    while (bigChunks.size > 0)
      Memory.free(bigChunks.pop())
    pool.decrementFreedBytes(totalChunkMemory)

    val freeBlocksOfSize = pool.freeBlocks(blockSize)
    freeBlocksOfSize.appendFrom(usedBlocks)
    usedBlocks.clear()

    if (currentBlock != 0)
      freeBlocksOfSize += currentBlock

    var j = 0
    while (j < references.size) {
      val rj = references(j)
      if (rj != null) {
        rj.release()
        references.update(j, null)
      }
      j += 1
    }
    references.clear()
  }

  private def free(): Unit = {
    if (blockSize != -1) {
      freeMemory()

      offsetWithinBlock = 0
      currentBlock = 0
      totalChunkMemory = 0
      blockSize = -1

      pool.reclaim(this)
    }
  }

  def getReferenceCount: Long = referenceCount

  def clear(): Unit = {
    assert(referenceCount == 1)
    assert(currentBlock != 0)

    val freeBlocksOfSize = pool.freeBlocks(blockSize)
    freeBlocksOfSize.appendFrom(usedBlocks)
    usedBlocks.clear()

    while (references.size > 0) {
      val rr = references.pop()
      if (rr != null)
        rr.release()
    }
    offsetWithinBlock = 0L
  }

  def release(): Unit = {
    assert(referenceCount > 0)
    referenceCount -= 1
    if (referenceCount == 0)
      free()
  }

  def reinitialize(newSize: Region.Size): Unit = {
    assert(blockSize == -1)
    assert(referenceCount == 0, referenceCount)
    blockSize = newSize
    blockByteSize = Region.SIZES(newSize)
    referenceCount = 1
    allocateNewBlock()
  }

  def close(): Unit = {
    free()
  }

  def numChunks: Int = bigChunks.size

  def numBlocks: Int = usedBlocks.size + (currentBlock != 0).toInt

  def currentOffset: Long = currentBlock + offsetWithinBlock

  def blockAddress: Long = currentBlock

  def addReferenceTo(r: RegionMemory): Unit = {
    references += r
    r.referenceCount += 1
  }

  def nReferencedRegions(): Long = references.size

  def setNumParents(n: Int): Unit = {
    assert(n >= references.size)
    references.setSizeUninitialized(n)
  }

  def setReferenceAtIndex(reference: RegionMemory, idx: Int): Unit = {
    reference.referenceCount += 1
    val existing = references(idx)
    if (existing != null)
      existing.release()
    references.update(idx, reference)
  }

  def getReferenceAtIndex(idx: Int, blockSize: Region.Size): RegionMemory = {
    val r = references(idx)
    if (r == null) {
      newParentReference(idx, blockSize)
    } else {
      assert(r.blockSize == blockSize)
      r.referenceCount += 1
      r
    }
  }

  def newParentReference(idx: Int, blockSize: Region.Size): RegionMemory = {
    val r = pool.getMemory(blockSize)
    setReferenceAtIndex(r, idx)
    r
  }

  def releaseReferenceAtIndex(idx: Int): Unit = {
    val r = references(idx)
    r.release()
    references.update(idx, null)
  }
}
