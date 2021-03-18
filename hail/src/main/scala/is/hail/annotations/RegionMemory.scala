package is.hail.annotations

import is.hail.expr.ir.{AnyRefArrayBuilder, LongArrayBuilder, LongMissingArrayBuilder}
import is.hail.types.physical.{PCanonicalNDArray, PNDArray}
import is.hail.utils._


final class RegionMemory(pool: RegionPool) extends AutoCloseable {
  private[this] val usedBlocks = new LongArrayBuilder(4)
  private[this] val bigChunks = new LongArrayBuilder(4)
  private[this] val jObjects = new AnyRefArrayBuilder[AnyRef](0)
  private val ndarrayRefs = new LongArrayBuilder(4)

  private[this] var totalChunkMemory = 0L
  private[this] var currentBlock: Long = 0L
  private[this] var offsetWithinBlock: Long = _
//  var stackTrace: Option[IndexedSeq[StackTraceElement]] = None

  // blockThreshold and blockByteSize are mutable because RegionMemory objects are reused with different sizes
  protected[annotations] var blockSize: Region.Size = -1
  private[this] var blockThreshold: Long = _
  private[this] var blockByteSize: Long = _

  private[this] val references = new AnyRefArrayBuilder[RegionMemory](4)
  private var referenceCount: Long = _

  def storeJavaObject(obj: AnyRef): Int = {
    val idx = jObjects.size
    jObjects.add(obj)
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
      usedBlocks.add(currentBlock)
    currentBlock = pool.getBlock(blockSize)
  }

  def getCurrentBlock(): Long = currentBlock

  private def allocateBigChunk(size: Long): Long = {
    val o = pool.getChunk(size)
    bigChunks.add(o)
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
    jObjects.clearAndSetNull()
  }

  private def freeFullBlocks(): Unit = freeFullBlocks(pool.freeBlocks(blockSize))

  private def freeFullBlocks(ab: LongArrayBuilder): Unit = {
    ab.appendFrom(usedBlocks)
    usedBlocks.clearAndResize()
  }

  private def releaseNDArrays(): Unit = {
    var i = 0
    while (i < ndarrayRefs.size) {
      val addr = this.ndarrayRefs(i)
      val curCount = PNDArray.getReferenceCount(addr)
      if (curCount == 1) {
        PNDArray.storeReferenceCount(addr, 0L)
        val bytesToFree = PNDArray.getByteSize(addr) + PNDArray.headerBytes
        pool.incrementAllocatedBytes(-bytesToFree)
        Memory.free(addr - PNDArray.headerBytes)
      } else {
        PNDArray.storeReferenceCount(addr, curCount - 1)
      }
      i += 1
    }

    this.ndarrayRefs.clear()
  }

  def getTotalChunkMemory(): Long = this.totalChunkMemory

  def totalManagedBytes(): Long = this.totalChunkMemory + usedBlocks.size * blockByteSize

  protected[annotations] def freeMemory(): Unit = {
    // freeMemory should be idempotent
    if (isFreed) {
      assert(references.size == 0)
      assert(usedBlocks.size == 0)
      assert(bigChunks.size == 0)
      assert(jObjects.size == 0)
      assert(ndarrayRefs.size == 0)
    } else {
      val freeBlocksOfSize = pool.freeBlocks(blockSize)
      if (currentBlock != 0)
        freeBlocksOfSize.add(currentBlock)

      freeFullBlocks(freeBlocksOfSize)
      freeChunks()
      freeObjects()
      releaseReferences()
      releaseNDArrays()

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
    releaseNDArrays()

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
    references.add(r)
    r.referenceCount += 1
  }

  def takeOwnershipOf(r: RegionMemory): Unit =
    references.add(r)

  def nReferencedRegions(): Long = references.size

  def setNumParents(n: Int): Unit = {
    assert(n >= references.size)
    references.setSizeUninitialized(n)
  }

  def setReferenceAtIndex(child: RegionMemory, idx: Int): Unit = {
    child.referenceCount += 1
    val existing = references(idx)
    if (existing != null)
      existing.release()
    references.update(idx, child)
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

  def allocateNDArray(size: Long): Long = {
    if (size <= 0L) {
      throw new IllegalArgumentException(s"Can't request ndarray of non-positive memory size, got ${size}")
    }

    val extra = PNDArray.headerBytes

    // This adjusted address is where the ndarray content starts
    val allocatedAddr = pool.getChunk(size + extra) + extra

    // The reference count and total size are stored just before the content.
    PNDArray.storeReferenceCount(allocatedAddr, 0L)
    PNDArray.storeByteSize(allocatedAddr, size)
    this.trackNDArray(allocatedAddr)
    allocatedAddr
  }

  def trackNDArray(alloc: Long): Unit = {
    this.ndarrayRefs.add(alloc)
    val curRefCount = Region.loadLong(alloc - 16)
    Region.storeLong(alloc - 16, curRefCount + 1L)
  }

  def listNDArrayRefs(): IndexedSeq[Long] = {
    this.ndarrayRefs.result().toIndexedSeq
  }
}
