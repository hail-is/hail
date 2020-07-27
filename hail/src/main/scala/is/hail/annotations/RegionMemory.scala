package is.hail.annotations

import is.hail.utils._

final class RegionMemory(pool: RegionPool) extends AutoCloseable {
  private val usedBlocks = new ArrayBuilder[Long](4)
  private val bigChunks = new ArrayBuilder[Long](4)
  private val jObjects = new ArrayBuilder[AnyRef](0)

  private var totalChunkMemory = 0L
  private var currentBlock: Long = 0L
  private var offsetWithinBlock: Long = _
//  var stackTrace: Option[IndexedSeq[StackTraceElement]] = None

  // blockThreshold and blockByteSize are mutable because RegionMemory objects are reused with different sizes
  protected[annotations] var blockSize: Region.Size = -1
  private var blockThreshold: Long = _
  private var blockByteSize: Long = _

  private val references = new ArrayBuilder[RegionMemory](4)
  private var referenceCount: Long = _

  def storeJavaObject(obj: AnyRef): Int = {
    val idx = jObjects.size
    jObjects += obj
    pool.addJavaObject()
    idx
  }

  def lookupJavaObject(idx: Int): AnyRef = {
    jObjects(idx)
  }

  def dumpMemoryInfo(): String = {
    s"""
       |Blocks Used = ${usedBlocks.size}, Chunks used = ${bigChunks.size}
       |Block Info:
       |  BlockSize = ${blockSize} ($blockByteSize bytes)
       |  Current Block Info:
       |    Current Block Address: ${currentBlock}
       |    Offset Within Block:   ${offsetWithinBlock}
       |  Used Blocks Info:
       |    BlockStarts: ${usedBlocks.result().toIndexedSeq}
       |""".stripMargin
  }

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
    o
  }

  def allocate(n: Long): Long = {
    val r = if (offsetWithinBlock + n <= blockByteSize) {
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
    r
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

  private def isFreed: Boolean = blockSize == -1

  private def freeChunks(): Unit = {
    pool.freeChunks(bigChunks, totalChunkMemory)
    bigChunks.clearAndResize()
    totalChunkMemory = 0L
  }

  private def releaseReferences(): Unit = {
    var j = 0
    while (j < references.size) {
      val rj = references(j)
      if (rj != null) {
        rj.release()
        references.update(j, null)
      }
      j += 1
    }
    references.clearAndResize()
  }

  private def freeObjects(): Unit = {
    pool.removeJavaObjects(jObjects.size)
    jObjects.clearAndSetMem(null)
  }

  private def freeFullBlocks(): Unit = freeFullBlocks(pool.freeBlocks(blockSize))

  private def freeFullBlocks(ab: ArrayBuilder[Long]): Unit = {
    ab.appendFrom(usedBlocks)
    usedBlocks.clearAndResize()
  }

  def getTotalChunkMemory(): Long = this.totalChunkMemory

  protected[annotations] def freeMemory(): Unit = {
    // freeMemory should be idempotent
    if (isFreed) {
      assert(references.size == 0)
      assert(usedBlocks.size == 0)
      assert(bigChunks.size == 0)
      assert(jObjects.size == 0)
    } else {
      val freeBlocksOfSize = pool.freeBlocks(blockSize)
      if (currentBlock != 0)
        freeBlocksOfSize += currentBlock

      freeFullBlocks(freeBlocksOfSize)
      freeChunks()
      freeObjects()
      releaseReferences()

      offsetWithinBlock = 0
      currentBlock = 0
      totalChunkMemory = 0
      blockSize = -1
    }
  }

  private def free(): Unit = {
    if (!isFreed) {
      freeMemory()
      pool.reclaim(this)
    }
//    stackTrace = None
  }

  def getReferenceCount: Long = referenceCount

  def clear(): Unit = {
    assert(referenceCount == 1)
    assert(currentBlock != 0)

    freeFullBlocks()
    freeChunks()
    freeObjects()
    releaseReferences()

    offsetWithinBlock = 0L
  }

  def release(): Unit = {
    assert(referenceCount > 0)
    referenceCount -= 1
    if (referenceCount == 0)
      free()
  }

  def initialize(newSize: Region.Size): Unit = {
    assert(blockSize == -1)
    assert(referenceCount == 0)
    assert(currentBlock == 0)
    assert(totalChunkMemory == 0)

//    this.stackTrace = Some(Thread.currentThread().getStackTrace.toIndexedSeq.drop(4))

    blockSize = newSize
    blockByteSize = Region.SIZES(blockSize)
    blockThreshold = Math.min(blockByteSize, Region.BLOCK_THRESHOLD)
    referenceCount = 1
    allocateNewBlock()
    offsetWithinBlock = 0L
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

  def takeOwnershipOf(r: RegionMemory): Unit =
    references += r

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
