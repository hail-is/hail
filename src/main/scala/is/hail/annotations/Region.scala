package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util

import is.hail.expr.types._
import is.hail.utils._
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object Region {
  def apply(): Region = {
    val blockSize = 16L * 1024L
    val blocks = new ArrayBuilder[Long]()
    blocks += Memory.malloc(blockSize)
    new Region(
      blocks,
      blockSize,
      0,
      0L,
      new ArrayBuilder()
    )
  }

  def scoped[T](f: Region => T): T =
    using(Region())(f)
}

final class Region private (
  private[this] var blocks: ArrayBuilder[Long],
  private[this] var blockSize: Long,
  private[this] var activeBlock: Int,
  private[this] var end: Long,
  private[this] var bigBlocks: ArrayBuilder[Long]
) extends UnKryoSerializable with AutoCloseable {
  def summary(): String = {
    s"""((${blocks.underlying.mkString(" ")})
        |$blockSize
        |$end
        |(${bigBlocks.underlying.mkString(" ")}))""".stripMargin
  }

  def capacity: Long = blockSize

  def copyFrom(other: Region, readStart: Long, writeStart: Long, n: Long) {
    assert(end <= capacity)
    assert(n >= 0)
    assert(readStart >= 0)
    assert(writeStart >= 0)
    Memory.memcpy(writeStart, readStart, n)
  }

  def loadInt(off: Long): Int = {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.loadInt(off)
  }

  def loadLong(off: Long): Long = {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.loadLong(off)
  }

  def loadFloat(off: Long): Float = {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.loadFloat(off)
  }

  def loadDouble(off: Long): Double = {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.loadDouble(off)
  }

  def loadAddress(off: Long): Long = {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.loadAddress(off)
  }

  def loadByte(off: Long): Byte = {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.loadByte(off)
  }

  def loadBytes(off: Long, n: Int): Array[Byte] = {
    assert(end <= capacity)
    assert(off >= 0)
    val a = new Array[Byte](n)
    Memory.memcpy(a, 0, off, n)
    a
  }

  def loadBytes(off: Long, to: Array[Byte], toOff: Long, n: Long) {
    assert(end <= capacity)
    assert(off >= 0)
    assert(toOff + n <= to.length)
    Memory.memcpy(to, toOff, off, n)
  }

  def storeInt(off: Long, i: Int) {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.storeInt(off, i)
  }

  def storeLong(off: Long, l: Long) {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.storeLong(off, l)
  }

  def storeFloat(off: Long, f: Float) {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.storeFloat(off, f)
  }

  def storeDouble(off: Long, d: Double) {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.storeDouble(off, d)
  }

  def storeAddress(off: Long, a: Long) {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.storeAddress(off, a)
  }

  def storeByte(off: Long, b: Byte) {
    assert(end <= capacity)
    assert(off >= 0)
    Memory.storeByte(off, b)
  }

  def storeBytes(off: Long, bytes: Array[Byte]) {
    storeBytes(off, bytes, 0, bytes.length)
  }

  def storeBytes(off: Long, bytes: Array[Byte], bytesOff: Long, n: Int) {
    assert(end <= capacity)
    assert(off >= 0)
    assert(bytesOff + n <= bytes.length)
    Memory.memcpy(off, bytes, bytesOff, n)
  }

  private def align(alignment: Long) {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    end = (end + (alignment - 1)) & ~(alignment - 1)
  }

  private def allocate(n: Long): Long = {
    assert(n >= 0)
    if (n > blockSize / 2) {
      // FIXME: is this guaranteed to be aligned to anything?
      log.info(s"Allocating a large block $n, currently have ${bigBlocks.length} large blocks")
      val mem = Memory.malloc(n)
      bigBlocks += mem
      mem
    } else {
      val required = end + n
      if (required > capacity) {
        // FIXME: is this guaranteed to be aligned to anything?
        activeBlock += 1
        val mem = if (activeBlock == blocks.length) {
          log.info(s"Allocating a new block, $end $n $capacity, currently have ${blocks.length} blocks")
          val temp = Memory.malloc(blockSize)
          blocks += temp
          temp
        } else {
          blocks(activeBlock)
        }
        end = n
        mem
      } else {
        val mem = blocks(activeBlock) + end
        end += n
        mem
      }
    }
  }

  def allocate(alignment: Long, n: Long): Long = {
    align(alignment)
    allocate(n)
  }

  def loadBoolean(off: Long): Boolean = {
    val b = loadByte(off)
    assert(b == 1 || b == 0)
    b != 0
  }

  def loadBit(byteOff: Long, bitOff: Long): Boolean = {
    val b = byteOff + (bitOff >> 3)
    (loadByte(b) & (1 << (bitOff & 7))) != 0
  }

  def setBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) | (1 << (bitOff & 7))).toByte)
  }

  def clearBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) & ~(1 << (bitOff & 7))).toByte)
  }

  def storeBit(byteOff: Long, bitOff: Long, b: Boolean) {
    if (b)
      setBit(byteOff, bitOff)
    else
      clearBit(byteOff, bitOff)
  }

  def appendInt(i: Int): Long = {
    val off = allocate(4, 4)
    storeInt(off, i)
    off
  }

  def appendBoolean(b: Boolean): Long =
    appendByte(b.toByte)

  def appendLong(l: Long): Long = {
    val off = allocate(8, 8)
    storeLong(off, l)
    off
  }

  def appendFloat(f: Float): Long = {
    val off = allocate(4, 4)
    storeFloat(off, f)
    off
  }

  def appendDouble(d: Double): Long = {
    val off = allocate(8, 8)
    storeDouble(off, d)
    off
  }

  def appendByte(b: Byte): Long = {
    val off = allocate(1)
    storeByte(off, b)
    off
  }

  def appendBinary(bytes: Array[Byte]): Long = {
    val n = bytes.length
    val off = allocate(math.max(TBinary.contentAlignment, 4), n + 4)
    storeInt(off, n)
    storeBytes(TBinary.bytesOffset(off), bytes)
    off
  }

  def appendBinarySlice(
    fromRegion: Region,
    fromOff: Long,
    start: Int,
    n: Int
  ): Long = {
    assert(n >= 0)
    val off = allocate(math.max(TBinary.contentAlignment, 4), n + 4)
    storeInt(off, n)
    copyFrom(fromRegion, TBinary.bytesOffset(fromOff) + start, off + 4, n)
    off
  }

  def appendString(s: String): Long =
    appendBinary(s.getBytes)

  def appendStringSlice(fromRegion: Region, fromOff: Long, start: Int, n: Int): Long =
    appendBinarySlice(fromRegion, fromOff, start, n)

  def appendArrayInt(a: Array[Int]): Long = {
    val t = TArray(+TInt32())
    val aoff = t.allocate(this, a.length)
    t.initialize(this, aoff, a.length)
    var i = 0
    while (i < a.length) {
      storeInt(t.elementOffset(aoff, a.length, i), a(i))
      i += 1
    }
    aoff
  }

  def clear() {
    end = 0
    activeBlock = 0
    bigBlocks.result().foreach(Memory.free)
    bigBlocks.clear()
  }

  def visit(t: Type, off: Long, v: ValueVisitor) {
    t match {
      case _: TBoolean => v.visitBoolean(loadBoolean(off))
      case _: TInt32 => v.visitInt32(loadInt(off))
      case _: TInt64 => v.visitInt64(loadLong(off))
      case _: TFloat32 => v.visitFloat32(loadFloat(off))
      case _: TFloat64 => v.visitFloat64(loadDouble(off))
      case _: TString =>
        val boff = off
        v.visitString(TString.loadString(this, boff))
      case _: TBinary =>
        val boff = off
        val length = TBinary.loadLength(this, boff)
        val b = loadBytes(TBinary.bytesOffset(boff), length)
        v.visitBinary(b)
      case t: TContainer =>
        val aoff = off
        val length = t.loadLength(this, aoff)
        v.enterArray(t, length)
        var i = 0
        while (i < length) {
          v.enterElement(i)
          if (t.isElementDefined(this, aoff, i))
            visit(t.elementType, t.loadElement(this, aoff, length, i), v)
          else
            v.visitMissing(t.elementType)
          i += 1
        }
        v.leaveArray()
      case t: TStruct =>
        v.enterStruct(t)
        var i = 0
        while (i < t.size) {
          val f = t.fields(i)
          v.enterField(f)
          if (t.isFieldDefined(this, off, i))
            visit(f.typ, t.loadField(this, off, i), v)
          else
            v.visitMissing(f.typ)
          v.leaveField()
          i += 1
        }
        v.leaveStruct()
      case t: TTuple =>
        v.enterTuple(t)
        var i = 0
        while (i < t.size) {
          v.enterElement(i)
          if (t.isFieldDefined(this, off, i))
            visit(t.types(i), t.loadField(this, off, i), v)
          else
            v.visitMissing(t.types(i))
          v.leaveElement()
          i += 1
        }
        v.leaveTuple()
      case t: ComplexType =>
        visit(t.representation, off, v)
    }
  }

  def pretty(t: Type, off: Long): String = {
    val v = new PrettyVisitor()
    visit(t, off, v)
    v.result()
  }

  def close(): Unit = {
    log.info(s"Freeing all blocks. ${blocks.length} ${bigBlocks.length}")
    blocks.result().foreach(Memory.free)
    bigBlocks.result().foreach(Memory.free)
  }

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }

  def prettyBits(): String = {
    "FIXME: implement prettyBits on Region"
  }
}
