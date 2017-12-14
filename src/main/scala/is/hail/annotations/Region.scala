package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{AltAllele, Locus, Variant}
import org.apache.spark.sql.Row

object Region {
  def apply(sizeHint: Long = 128): Region = {
    new Region(new Array[Byte](sizeHint.toInt))
  }
}

final class Region(private var mem: Array[Byte], private var end: Long = 0) extends KryoSerializable with Serializable {
  def size: Long = end

  def capacity: Long = mem.length

  def copyFrom(other: Region, readStart: Long, writeStart: Long, n: Long) {
    assert(size <= capacity)
    assert(other.size <= other.capacity)
    assert(n >= 0)
    assert(readStart >= 0 && readStart + n <= other.size)
    assert(writeStart >= 0 && writeStart + n <= size)
    Memory.memcpy(mem, writeStart, other.mem, readStart, n)
  }

  def loadInt(off: Long): Int = {
    assert(size <= capacity)
    assert(off >= 0 && off + 4 <= size)
    Memory.loadInt(mem, off)
  }

  def loadLong(off: Long): Long = {
    assert(size <= capacity)
    assert(off >= 0 && off + 8 <= size)
    Memory.loadLong(mem, off)
  }

  def loadFloat(off: Long): Float = {
    assert(size <= capacity)
    assert(off >= 0 && off + 4 <= size)
    Memory.loadFloat(mem, off)
  }

  def loadDouble(off: Long): Double = {
    assert(size <= capacity)
    assert(off >= 0 && off + 8 <= size)
    Memory.loadDouble(mem, off)
  }

  def loadAddress(off: Long): Long = {
    assert(size <= capacity)
    assert(off >= 0 && off + 8 <= size)
    Memory.loadAddress(mem, off)
  }

  def loadByte(off: Long): Byte = {
    assert(size <= capacity)
    assert(off >= 0 && off + 1 <= size)
    Memory.loadByte(mem, off)
  }

  def loadBytes(off: Long, n: Int): Array[Byte] = {
    assert(size <= capacity)
    assert(off >= 0 && off + n <= size)
    val a = new Array[Byte](n)
    Memory.memcpy(a, 0, mem, off, n)
    a
  }

  def loadBytes(off: Long, to: Array[Byte], toOff: Long, n: Long) {
    assert(size <= capacity)
    assert(off >= 0 && off + n <= size)
    assert(toOff + n <= to.length)
    Memory.memcpy(to, toOff, mem, off, n)
  }

  def storeInt(off: Long, i: Int) {
    assert(size <= capacity)
    assert(off >= 0 && off + 4 <= size)
    Memory.storeInt(mem, off, i)
  }

  def storeLong(off: Long, l: Long) {
    assert(size <= capacity)
    assert(off >= 0 && off + 8 <= size)
    Memory.storeLong(mem, off, l)
  }

  def storeFloat(off: Long, f: Float) {
    assert(size <= capacity)
    assert(off >= 0 && off + 4 <= size)
    Memory.storeFloat(mem, off, f)
  }

  def storeDouble(off: Long, d: Double) {
    assert(size <= capacity)
    assert(off >= 0 && off + 8 <= size)
    Memory.storeDouble(mem, off, d)
  }

  def storeAddress(off: Long, a: Long) {
    assert(size <= capacity)
    assert(off >= 0 && off + 8 <= size)
    Memory.storeAddress(mem, off, a)
  }

  def storeByte(off: Long, b: Byte) {
    assert(size <= capacity)
    assert(off >= 0 && off + 1 <= size)
    Memory.storeByte(mem, off, b)
  }

  def storeBytes(off: Long, bytes: Array[Byte]) {
    storeBytes(off, bytes, 0, bytes.length)
  }

  def storeBytes(off: Long, bytes: Array[Byte], bytesOff: Long, n: Int) {
    assert(size <= capacity)
    assert(off >= 0 && off + n <= size)
    assert(bytesOff + n <= bytes.length)
    Memory.memcpy(mem, off, bytes, bytesOff, n)
  }

  def ensure(n: Long) {
    val required = size + n
    if (capacity < required) {
      val newLength = (capacity * 2).max(required)
      mem = util.Arrays.copyOf(mem, newLength.toInt)
    }
  }

  def align(alignment: Long) {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    end = (end + (alignment - 1)) & ~(alignment - 1)
  }

  def allocate(n: Long): Long = {
    assert(n >= 0)
    val off = end
    ensure(n)
    end += n
    off
  }

  def alignAndAllocate(n: Long): Long = {
    align(n)
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
    val off = alignAndAllocate(4)
    storeInt(off, i)
    off
  }

  def appendLong(l: Long): Long = {
    val off = alignAndAllocate(8)
    storeLong(off, l)
    off
  }

  def appendFloat(f: Float): Long = {
    val off = alignAndAllocate(4)
    storeFloat(off, f)
    off
  }

  def appendDouble(d: Double): Long = {
    val off = alignAndAllocate(8)
    storeDouble(off, d)
    off
  }

  def appendByte(b: Byte): Long = {
    val off = allocate(1)
    storeByte(off, b)
    off
  }

  def appendBytes(bytes: Array[Byte]): Long = {
    val off = allocate(bytes.length)
    storeBytes(off, bytes)
    off
  }

  def appendBytes(bytes: Array[Byte], bytesOff: Long, n: Int): Long = {
    assert(bytesOff + n <= bytes.length)
    val off = allocate(n)
    storeBytes(off, bytes, bytesOff, n)
    off
  }

  def appendBinary(bytes: Array[Byte]): Long = {
    align(TBinary.contentAlignment)
    val startOff = appendInt(bytes.length)
    appendBytes(bytes)
    startOff
  }

  def appendString(s: String): Long =
    appendBinary(s.getBytes)

  def appendArrayInt(a: Array[Int]): Long = {
    val off = appendInt(a.length)
    var i = 0
    while (i < a.length) {
      appendInt(a(i))
      i += 1
    }
    off
  }

  def clear(newEnd: Long) {
    assert(newEnd <= end)
    end = newEnd
  }

  def clear() {
    end = 0
  }

  def setFrom(from: Region) {
    if (from.end > capacity) {
      val newLength = math.max((capacity * 3) / 2, from.end)
      mem = new Array[Byte](newLength.toInt)
    }
    Memory.memcpy(mem, 0, from.mem, 0, from.end)
    end = from.end
  }

  def copy(): Region = {
    new Region(util.Arrays.copyOf(mem, end.toInt), end)
  }

  override def write(kryo: Kryo, output: Output) {
    output.writeLong(end)

    assert(end <= Int.MaxValue)
    val smallEnd = end.toInt
    output.write(mem, 0, smallEnd)
  }

  override def read(kryo: Kryo, input: Input) {
    end = input.readLong()
    assert(end <= Int.MaxValue)
    val smallEnd = end.toInt
    mem = new Array[Byte](smallEnd)
    input.read(mem)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeLong(end)

    assert(end <= Int.MaxValue)
    val smallEnd = end.toInt
    out.write(mem, 0, smallEnd)
  }

  private def readObject(in: ObjectInputStream) {
    end = in.readLong()
    assert(end <= Int.MaxValue)
    val smallOffset = end.toInt
    mem = new Array[Byte](smallOffset)
    in.read(mem)
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
      case t: ComplexType =>
        visit(t.representation, off, v)
    }
  }

  def pretty(t: Type, off: Long): String = {
    val v = new PrettyVisitor()
    visit(t, off, v)
    v.result()
  }
}
