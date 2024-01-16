package is.hail.annotations

import is.hail.asm4s
import is.hail.asm4s.Code
import is.hail.types.physical._
import is.hail.utils._

object Region {
  type Size = Int
  val REGULAR: Size = 0
  val SMALL: Size = 1
  val TINY: Size = 2
  val TINIER: Size = 3

  val SIZES: Array[Long] = Array(64 * 1024, 8 * 1024, 1024, 256)
  val BLOCK_THRESHOLD: Long = 4 * 1024

  def loadInt(addr: Long): Int = Memory.loadInt(addr)

  def loadLong(addr: Long): Long = Memory.loadLong(addr)

  def loadFloat(addr: Long): Float = Memory.loadFloat(addr)

  def loadDouble(addr: Long): Double = Memory.loadDouble(addr)

  def loadAddress(addr: Long): Long = Memory.loadLong(addr)

  def loadByte(addr: Long): Byte = Memory.loadByte(addr)

  def loadShort(addr: Long): Short = Memory.loadShort(addr)

  def loadChar(addr: Long): Char = Memory.loadShort(addr).toChar

  def storeInt(addr: Long, v: Int): Unit = Memory.storeInt(addr, v)

  def storeLong(addr: Long, v: Long): Unit = Memory.storeLong(addr, v)

  def storeFloat(addr: Long, v: Float): Unit = Memory.storeFloat(addr, v)

  def storeDouble(addr: Long, v: Double): Unit = Memory.storeDouble(addr, v)

  def storeAddress(addr: Long, v: Long): Unit = Memory.storeAddress(addr, v)

  def storeByte(addr: Long, v: Byte): Unit = Memory.storeByte(addr, v)

  def storeShort(addr: Long, s: Short): Unit = Memory.storeShort(addr, s)

  def storeChar(addr: Long, c: Char): Unit = Memory.storeShort(addr, c.toShort)

  def loadBoolean(addr: Long): Boolean = if (Memory.loadByte(addr) == 0) false else true

  def storeBoolean(addr: Long, v: Boolean): Unit = Memory.storeByte(addr, if (v) 1 else 0)

  def loadBytes(addr: Long, n: Int): Array[Byte] = {
    val a = new Array[Byte](n)
    Memory.copyToArray(a, 0, addr, n)
    a
  }

  def loadBytes(addr: Long, dst: Array[Byte], dstOff: Long, n: Long): Unit =
    Memory.copyToArray(dst, dstOff, addr, n)

  def storeBytes(addr: Long, src: Array[Byte]): Unit =
    Memory.copyFromArray(addr, src, 0, src.length)

  def storeBytes(addr: Long, src: Array[Byte], srcOff: Long, n: Long): Unit =
    Memory.copyFromArray(addr, src, srcOff, n)

  def copyFrom(srcOff: Long, dstOff: Long, n: Long): Unit =
    Memory.memcpy(dstOff, srcOff, n)

  def setMemory(offset: Long, size: Long, b: Byte): Unit = {
    if (size < (1 << 10)) {
      var x = 0L
      while (x < size) {
        Memory.storeByte(offset + x, b)
        x += 1
      }
    } else {
      Memory.memset(offset, size, b)
    }
  }

  def loadBit(byteOff: Long, bitOff: Long): Boolean = {
    val b = byteOff + (bitOff >> 3)
    (loadByte(b) & (1 << (bitOff & 7).toInt)) != 0
  }

  def setBit(byteOff: Long, bitOff: Long): Unit = {
    val b = byteOff + (bitOff >> 3)
    storeByte(b, (loadByte(b) | (1 << (bitOff & 7).toInt)).toByte)
  }

  def clearBit(byteOff: Long, bitOff: Long): Unit = {
    val b = byteOff + (bitOff >> 3)
    storeByte(b, (loadByte(b) & ~(1 << (bitOff & 7).toInt)).toByte)
  }

  def storeBit(byteOff: Long, bitOff: Long, b: Boolean): Unit = {
    if (b)
      setBit(byteOff, bitOff)
    else
      clearBit(byteOff, bitOff)
  }

  def loadInt(addr: Code[Long]): Code[Int] =
    Code.invokeScalaObject1[Long, Int](Region.getClass, "loadInt", addr)

  def loadLong(addr: Code[Long]): Code[Long] =
    Code.invokeScalaObject1[Long, Long](Region.getClass, "loadLong", addr)

  def loadFloat(addr: Code[Long]): Code[Float] =
    Code.invokeScalaObject1[Long, Float](Region.getClass, "loadFloat", addr)

  def loadDouble(addr: Code[Long]): Code[Double] =
    Code.invokeScalaObject1[Long, Double](Region.getClass, "loadDouble", addr)

  def loadAddress(addr: Code[Long]): Code[Long] =
    Code.invokeScalaObject1[Long, Long](Region.getClass, "loadAddress", addr)

  def loadByte(addr: Code[Long]): Code[Byte] =
    Code.invokeScalaObject1[Long, Byte](Region.getClass, "loadByte", addr)

  def loadShort(addr: Code[Long]): Code[Short] =
    Code.invokeScalaObject1[Long, Short](Region.getClass, "loadShort", addr)

  def loadChar(addr: Code[Long]): Code[Char] =
    Code.invokeScalaObject1[Long, Char](Region.getClass, "loadChar", addr)

  def storeInt(addr: Code[Long], v: Code[Int]): Code[Unit] =
    Code.invokeScalaObject2[Long, Int, Unit](Region.getClass, "storeInt", addr, v)

  def storeLong(addr: Code[Long], v: Code[Long]): Code[Unit] =
    Code.invokeScalaObject2[Long, Long, Unit](Region.getClass, "storeLong", addr, v)

  def storeFloat(addr: Code[Long], v: Code[Float]): Code[Unit] =
    Code.invokeScalaObject2[Long, Float, Unit](Region.getClass, "storeFloat", addr, v)

  def storeDouble(addr: Code[Long], v: Code[Double]): Code[Unit] =
    Code.invokeScalaObject2[Long, Double, Unit](Region.getClass, "storeDouble", addr, v)

  def storeChar(addr: Code[Long], v: Code[Char]): Code[Unit] =
    Code.invokeScalaObject2[Long, Char, Unit](Region.getClass, "storeChar", addr, v)

  def storeAddress(addr: Code[Long], v: Code[Long]): Code[Unit] =
    Code.invokeScalaObject2[Long, Long, Unit](Region.getClass, "storeAddress", addr, v)

  def storeByte(addr: Code[Long], v: Code[Byte]): Code[Unit] =
    Code.invokeScalaObject2[Long, Byte, Unit](Region.getClass, "storeByte", addr, v)

  def storeShort(addr: Code[Long], v: Code[Short]): Code[Unit] =
    Code.invokeScalaObject2[Long, Short, Unit](Region.getClass, "storeShort", addr, v)

  def loadBoolean(addr: Code[Long]): Code[Boolean] =
    Code.invokeScalaObject1[Long, Boolean](Region.getClass, "loadBoolean", addr)

  def storeBoolean(addr: Code[Long], v: Code[Boolean]): Code[Unit] =
    Code.invokeScalaObject2[Long, Boolean, Unit](Region.getClass, "storeBoolean", addr, v)

  def loadBytes(addr: Code[Long], n: Code[Int]): Code[Array[Byte]] =
    Code.invokeScalaObject2[Long, Int, Array[Byte]](Region.getClass, "loadBytes", addr, n)

  def loadBytes(addr: Code[Long], dst: Code[Array[Byte]], dstOff: Code[Long], n: Code[Long])
    : Code[Unit] =
    Code.invokeScalaObject4[Long, Array[Byte], Long, Long, Unit](
      Region.getClass,
      "loadBytes",
      addr,
      dst,
      dstOff,
      n,
    )

  def storeBytes(addr: Code[Long], src: Code[Array[Byte]]): Code[Unit] =
    Code.invokeScalaObject2[Long, Array[Byte], Unit](Region.getClass, "storeBytes", addr, src)

  def storeBytes(addr: Code[Long], src: Code[Array[Byte]], srcOff: Code[Long], n: Code[Long])
    : Code[Unit] =
    Code.invokeScalaObject4[Long, Array[Byte], Long, Long, Unit](
      Region.getClass,
      "storeBytes",
      addr,
      src,
      srcOff,
      n,
    )

  def copyFrom(srcOff: Code[Long], dstOff: Code[Long], n: Code[Long]): Code[Unit] =
    Code.invokeScalaObject3[Long, Long, Long, Unit](Region.getClass, "copyFrom", srcOff, dstOff, n)

  def loadBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Boolean] =
    Code.invokeScalaObject2[Long, Long, Boolean](Region.getClass, "loadBit", byteOff, bitOff)

  def setBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] =
    Code.invokeScalaObject2[Long, Long, Unit](Region.getClass, "setBit", byteOff, bitOff)

  def clearBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] =
    Code.invokeScalaObject2[Long, Long, Unit](Region.getClass, "clearBit", byteOff, bitOff)

  def storeBit(byteOff: Code[Long], bitOff: Code[Long], b: Code[Boolean]): Code[Unit] =
    Code.invokeScalaObject3[Long, Long, Boolean, Unit](
      Region.getClass,
      "storeBit",
      byteOff,
      bitOff,
      b,
    )

  def setMemory(offset: Code[Long], size: Code[Long], b: Code[Byte]): Code[Unit] =
    Code.invokeScalaObject3[Long, Long, Byte, Unit](Region.getClass, "setMemory", offset, size, b)

  def containsNonZeroBits(address: Code[Long], nBits: Code[Long]): Code[Boolean] =
    Code.invokeScalaObject2[Long, Long, Boolean](
      Region.getClass,
      "containsNonZeroBits",
      address,
      nBits,
    )

  def containsNonZeroBits(address: Long, nBits: Long): Boolean = {
    assert((address & 0x3) == 0)

    var bitsRead: Long = 0

    if ((address & 0x7) != 0 && nBits >= 32) {
      if (loadInt(address) != 0)
        return true

      bitsRead += 32
    }

    while (nBits - bitsRead >= 64) {
      if (loadLong(address + bitsRead / 8) != 0)
        return true

      bitsRead += 64
    }

    while (nBits - bitsRead >= 32) {
      if (loadInt(address + bitsRead / 8) != 0)
        return true

      bitsRead += 32
    }

    while (nBits - bitsRead >= 8) {
      if (loadByte(address + bitsRead / 8) != 0)
        return true

      bitsRead += 8
    }

    while (bitsRead < nBits) {
      if (loadBit(address, bitsRead))
        return true

      bitsRead += 1
    }

    false
  }

  val sharedChunkHeaderBytes = 16L
  def getSharedChunkRefCount(ndAddr: Long): Long = Region.loadLong(ndAddr - sharedChunkHeaderBytes)

  def storeSharedChunkRefCount(ndAddr: Long, newCount: Long): Unit =
    Region.storeLong(ndAddr - sharedChunkHeaderBytes, newCount)

  def getSharedChunkByteSize(ndAddr: Long): Long = Region.loadLong(ndAddr - 8L)
  def getSharedChunkByteSize(ndAddr: Code[Long]): Code[Long] = Region.loadLong(ndAddr - 8L)

  def storeSharedChunkByteSize(ndAddr: Long, byteSize: Long): Unit =
    Region.storeLong(ndAddr - 8L, byteSize)

  def stagedCreate(blockSize: Size, pool: Code[RegionPool]): Code[Region] =
    Code.invokeScalaObject2[Int, RegionPool, Region](
      Region.getClass,
      "apply",
      asm4s.const(blockSize),
      pool,
    )

  def apply(blockSize: Region.Size = Region.REGULAR, pool: RegionPool): Region =
    pool.getRegion(blockSize)

  def pretty(off: Long, n: Int, header: String): String = {
    val linewidth = 4
    s"$header\n" +
      Region.loadBytes(off, n)
        .map(b => "%02x".format(b)).grouped(8).map(_.mkString(" "))
        .grouped(linewidth).zipWithIndex
        .map { case (s, i) =>
          "    %016x  ".format(off + (8 * 8 * linewidth * i)) + s.mkString("  ")
        }
        .mkString("\n")
  }

  def pretty(off: Code[Long], n: Int, header: Code[String]): Code[String] =
    Code.invokeScalaObject3[Long, Int, String, String](
      Region.getClass,
      "pretty",
      off,
      asm4s.const(n),
      header,
    )

  def pretty(t: PType, off: Long): String = {
    val v = new PrettyVisitor()
    visit(t, off, v)
    v.result()
  }

  def visit(t: PType, off: Long, v: ValueVisitor): Unit = {
    t match {
      case _: PBoolean => v.visitBoolean(Region.loadBoolean(off))
      case _: PInt32 => v.visitInt32(Region.loadInt(off))
      case _: PInt64 => v.visitInt64(Region.loadLong(off))
      case _: PFloat32 => v.visitFloat32(Region.loadFloat(off))
      case _: PFloat64 => v.visitFloat64(Region.loadDouble(off))
      case t: PString =>
        v.visitString(t.loadString(off))
      case t: PBinary =>
        val b = t.loadBytes(off)
        v.visitBinary(b)
      case t: PContainer =>
        val aoff = off
        val pt = t
        val length = pt.loadLength(aoff)
        v.enterArray(t, length)
        var i = 0
        while (i < length) {
          v.enterElement(i)
          if (pt.isElementDefined(aoff, i))
            visit(t.elementType, pt.loadElement(aoff, length, i), v)
          else
            v.visitMissing(t.elementType)
          i += 1
        }
        v.leaveArray()
      case t: PStruct =>
        v.enterStruct(t)
        var i = 0
        while (i < t.size) {
          val f = t.fields(i)
          v.enterField(f)
          if (t.isFieldDefined(off, i))
            visit(f.typ, t.loadField(off, i), v)
          else
            v.visitMissing(f.typ)
          v.leaveField()
          i += 1
        }
        v.leaveStruct()
      case t: PTuple =>
        v.enterTuple(t)
        var i = 0
        while (i < t.size) {
          v.enterElement(i)
          if (t.isFieldDefined(off, i))
            visit(t.types(i), t.loadField(off, i), v)
          else
            v.visitMissing(t.types(i))
          v.leaveElement()
          i += 1
        }
        v.leaveTuple()
      case t: PCanonicalInterval =>
        visit(t.representation, off, v)
      case t: PCanonicalLocus =>
        visit(t.representation, off, v)
      case t: PCanonicalCall =>
        visit(t.representation, off, v)
      case t: PCanonicalNDArray =>
        visit(t.representation, off, v)
    }
  }
}

final class Region protected[annotations] (
  var blockSize: Region.Size,
  var pool: RegionPool,
  var memory: RegionMemory = null,
) extends AutoCloseable {
  def getMemory(): RegionMemory = memory

  def isValid(): Boolean = memory != null

  def allocate(n: Long): Long = {
    assert(n >= 0)
    memory.allocate(n)
  }

  def allocate(a: Long, n: Long): Long = {
    assert(n >= 0)
    memory.allocate(a, n)
  }

  def invalidate(): Unit =
    if (memory != null) {
      memory.release()
      memory = null
    }

  def clear(): Unit = {
    if (memory.getReferenceCount == 1) {
      memory.clear()
    } else {
      memory.release()
      memory = pool.getMemory(blockSize)
    }
  }

  def allocateSharedChunk(nBytes: Long): Long = {
    assert(nBytes >= 0L)
    memory.allocateSharedChunk(nBytes)
  }

  def trackSharedChunk(addr: Long): Unit =
    memory.trackSharedChunk(addr)

  def close(): Unit =
    invalidate()

  def addReferenceTo(r: Region): Unit =
    memory.addReferenceTo(r.memory)

  def move(r: Region): Unit = {
    r.memory.takeOwnershipOf(memory)
    memory = pool.getMemory(blockSize)
  }

  def nReferencedRegions(): Long = memory.nReferencedRegions()

  def getNewRegion(blockSize: Region.Size): Unit = {
    if (memory != null)
      memory.release()
    memory = pool.getMemory(blockSize)
  }

  def setNumParents(n: Int): Unit =
    memory.setNumParents(n)

  def setParentReference(child: Region, idx: Int): Unit =
    memory.setReferenceAtIndex(child.memory, idx)

  def getReferenceCount(): Long = memory.getReferenceCount

  def getParentReference(idx: Int, blockSize: Region.Size): Region =
    new Region(blockSize, pool, memory.getReferenceAtIndex(idx, blockSize))

  def setFromParentReference(r: Region, idx: Int, blockSize: Region.Size): Unit = {
    invalidate()
    memory = r.memory.getReferenceAtIndex(idx, blockSize)
    this.blockSize = blockSize
  }

  def newParentReference(idx: Int, blockSize: Region.Size): Region = {
    val r = pool.getRegion(blockSize)
    setParentReference(r, idx)
    r
  }

  def unreferenceRegionAtIndex(idx: Int): Unit =
    memory.releaseReferenceAtIndex(idx)

  def storeJavaObject(obj: AnyRef): Int = memory.storeJavaObject(obj)

  def lookupJavaObject(idx: Int): AnyRef = memory.lookupJavaObject(idx)

  def prettyBits(): String =
    "FIXME: implement prettyBits on Region"

  def getPool(): RegionPool =
    pool

  def totalManagedBytes(): Long = memory.totalManagedBytes()
}

object RegionUtils {
  def printAddr(off: Long, name: String): String = s"$name: ${"%016x".format(off)}"

  def printAddr(off: Code[Long], name: String): Code[String] =
    Code.invokeScalaObject2[Long, String, String](RegionUtils.getClass, "printAddr", off, name)

  def printBytes(off: Long, n: Int, header: String): String =
    Region.loadBytes(off, n).zipWithIndex
      .grouped(16)
      .map(bs =>
        bs.map { case (b, _) => "%02x".format(b) }.mkString(
          "  %016x: ".format(off + bs(0)._2),
          " ",
          "",
        )
      )
      .mkString(if (header != null) s"$header\n" else "\n", "\n", "")

  def printBytes(off: Code[Long], n: Int, header: String): Code[String] =
    Code.invokeScalaObject3[Long, Int, String, String](
      RegionUtils.getClass,
      "printBytes",
      off,
      n,
      asm4s.const(header),
    )

  def logRegionStats(header: String, region: RegionMemory): Unit = {
    val size = region.blockSize
    val nUsed = region.numBlocks
    val off = region.currentOffset
    val numChunks = region.numChunks
    val ndarrays = region.numNDArrays
    val addr = "%016x".format(region.getCurrentBlock())

    val nReferenced = region.nReferencedRegions()

    info(
      s"""
         |$header:
         |  block size: $size
         | blocks used: $nUsed
         | current off: $off
         |  big chunks: $numChunks
         |  ndarrays: $ndarrays
         |  block addr: $addr
         |  referenced: $nReferenced
       """.stripMargin
    )
  }

  def logRegionStats(header: String, region: Code[Region]): Code[Unit] =
    Code.invokeScalaObject2[String, RegionMemory, Unit](
      RegionUtils.getClass,
      "logRegionStats",
      header,
      region.invoke[RegionMemory]("getMemory"),
    )
}
