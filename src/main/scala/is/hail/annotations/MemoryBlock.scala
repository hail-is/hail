package is.hail.annotations

import java.util

import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{AltAllele, Genotype, Locus, Variant}
import org.apache.spark.sql.Row
import org.apache.spark.unsafe.Platform

import scala.collection.mutable

final class MemoryBlock(val mem: Array[Long]) {
  require(mem.length < (Integer.MAX_VALUE / 8), "too big")

  def sizeInBytes: Int = mem.length * 8

  def copyFrom(other: MemoryBlock, readStart: Int, writeStart: Int, size: Int) {
    Platform.copyMemory(other.mem, readStart + Platform.LONG_ARRAY_OFFSET, mem,
      writeStart + Platform.LONG_ARRAY_OFFSET, size)
  }

  def loadInt(off: Int): Int = {
    assert(off + 4 <= sizeInBytes, s"tried to read int from offset $off with array size $sizeInBytes")
    Platform.getInt(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadLong(off: Int): Long = {
    assert(off + 8 <= sizeInBytes, s"tried to read long from offset $off with array size $sizeInBytes")
    Platform.getLong(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadFloat(off: Int): Float = {
    assert(off + 4 <= sizeInBytes, s"tried to read float from offset $off with array size $sizeInBytes")
    Platform.getFloat(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadDouble(off: Int): Double = {
    assert(off + 8 <= sizeInBytes, s"tried to read double from offset $off with array size $sizeInBytes")
    Platform.getDouble(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadByte(off: Int): Byte = {
    assert(off + 1 <= sizeInBytes, s"tried to read byte from offset $off with array size $sizeInBytes")
    Platform.getByte(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadBytes(off: Int, size: Int): Array[Byte] = {
    assert(off + size <= sizeInBytes, s"tried to read bytes of size $size from offset $off with array size $sizeInBytes")
    val a = new Array[Byte](size)
    Platform.copyMemory(mem, Platform.LONG_ARRAY_OFFSET + off, a, Platform.BYTE_ARRAY_OFFSET, size)
    a
  }

  def loadBytes(off: Int, size: Int, a: Array[Byte]) {
    assert(off + size <= sizeInBytes, s"tried to read bytes of size $size from offset $off with array size $sizeInBytes")
    assert(a.length >= size)
    Platform.copyMemory(mem, Platform.LONG_ARRAY_OFFSET + off, a, Platform.BYTE_ARRAY_OFFSET, size)
  }

  def storeInt(off: Int, i: Int) {
    assert(off + 4 <= sizeInBytes, s"tried to store int to offset $off with array size $sizeInBytes")
    Platform.putInt(mem, Platform.LONG_ARRAY_OFFSET + off, i)
  }

  def storeLong(off: Int, l: Long) {
    assert(off + 8 <= sizeInBytes, s"tried to store long to offset $off with array size $sizeInBytes")
    Platform.putLong(mem, Platform.LONG_ARRAY_OFFSET + off, l)
  }

  def storeFloat(off: Int, f: Float) {
    assert(off + 4 <= sizeInBytes, s"tried to store float to offset $off with array size $sizeInBytes")
    Platform.putFloat(mem, Platform.LONG_ARRAY_OFFSET + off, f)
  }

  def storeDouble(off: Int, d: Double) {
    assert(off + 8 <= sizeInBytes, s"tried to store double to offset $off with array size $sizeInBytes")
    Platform.putDouble(mem, Platform.LONG_ARRAY_OFFSET + off, d)
  }

  def storeByte(off: Int, b: Byte) {
    assert(off + 1 <= sizeInBytes, s"tried to store byte to offset $off with array size $sizeInBytes")
    Platform.putByte(mem, Platform.LONG_ARRAY_OFFSET + off, b)
  }

  def storeBytes(off: Int, bytes: Array[Byte]) {
    assert(off + bytes.length <= sizeInBytes, s"tried to store ${ bytes.length } bytes to offset $off with array size $sizeInBytes")
    Platform.copyMemory(bytes, Platform.BYTE_ARRAY_OFFSET, mem, Platform.LONG_ARRAY_OFFSET + off, bytes.length)
  }

  def storeBytes(off: Int, bytes: Array[Byte], size: Int) {
    assert(off + size <= sizeInBytes, s"tried to store ${ bytes.length } bytes to offset $off with array size $sizeInBytes")
    Platform.copyMemory(bytes, Platform.BYTE_ARRAY_OFFSET, mem, Platform.LONG_ARRAY_OFFSET + off, size)
  }

  def reallocate(size: Int): MemoryBlock = {
    if (sizeInBytes < size) {
      val newMem = new Array[Long](math.max(mem.length * 2, (size + 7) / 8))
      Platform.copyMemory(mem, Platform.LONG_ARRAY_OFFSET, newMem, Platform.LONG_ARRAY_OFFSET, sizeInBytes)
      new MemoryBlock(newMem)
    } else
      this
  }

  def copy(): MemoryBlock = copy(mem.length)

  def copy(length: Int): MemoryBlock = new MemoryBlock(util.Arrays.copyOf(mem, length))
}

object MemoryBuffer {
  def apply(sizeHint: Int = 128): MemoryBuffer = {
    new MemoryBuffer(new MemoryBlock(new Array[Long]((sizeHint + 7) / 8)))
  }
}

final class MemoryBuffer(var mb: MemoryBlock) {
  var offset: Int = 0

  def sizeInBytes: Int = offset

  def alignAndEnsure(size: Int) {
    align(size)
    ensure(size)
  }

  def ensure(size: Int) {
    mb = mb.reallocate(offset + size)
  }

  def loadInt(off: Int): Int = mb.loadInt(off)

  def loadLong(off: Int): Long = mb.loadLong(off)

  def loadFloat(off: Int): Float = mb.loadFloat(off)

  def loadDouble(off: Int): Double = mb.loadDouble(off)

  def loadByte(off: Int): Byte = mb.loadByte(off)

  def loadBytes(off: Int, size: Int): Array[Byte] = mb.loadBytes(off, size)

  def loadBytes(off: Int, size: Int, a: Array[Byte]) {
    mb.loadBytes(off, size, a)
  }

  def loadBit(byteOff: Int, bitOff: Int): Boolean = {
    val b = byteOff + (bitOff >> 3)
    (mb.loadByte(b) & (1 << (bitOff & 7))) != 0
  }

  def storeInt(off: Int, i: Int) {
    mb.storeInt(off, i)
  }

  def storeLong(off: Int, l: Long) {
    mb.storeLong(off, l)
  }

  def storeFloat(off: Int, f: Float) {
    mb.storeFloat(off, f)
  }

  def storeDouble(off: Int, d: Double) {
    mb.storeDouble(off, d)
  }

  def storeByte(off: Int, b: Byte) {
    mb.storeByte(off, b)
  }

  def storeBytes(off: Int, bytes: Array[Byte]) {
    mb.storeBytes(off, bytes)
  }

  def setBit(byteOff: Int, bitOff: Int) {
    val b = byteOff + (bitOff >> 3)
    mb.storeByte(b,
      (mb.loadByte(b) | (1 << (bitOff & 7))).toByte)
  }

  def clearBit(byteOff: Int, bitOff: Int) {
    val b = byteOff + (bitOff >> 3)
    mb.storeByte(b,
      (mb.loadByte(b) & ~(1 << (bitOff & 7))).toByte)
  }

  def storeBit(byteOff: Int, bitOff: Int, b: Boolean) {
    if (b)
      setBit(byteOff, bitOff)
    else
      clearBit(byteOff, bitOff)
  }

  def appendInt(i: Int) {
    alignAndEnsure(4)
    mb.storeInt(offset, i)
    offset += 4
  }

  def appendLong(l: Long) {
    alignAndEnsure(8)
    mb.storeLong(offset, l)
    offset += 8
  }

  def appendFloat(f: Float) {
    alignAndEnsure(4)
    mb.storeFloat(offset, f)
    offset += 4
  }

  def appendDouble(d: Double) {
    alignAndEnsure(8)
    mb.storeDouble(offset, d)
    offset += 8
  }

  def appendByte(b: Byte) {
    ensure(1)
    mb.storeByte(offset, b)
    offset += 1
  }

  def appendBytes(bytes: Array[Byte]) {
    ensure(bytes.length)
    mb.storeBytes(offset, bytes)
    offset += bytes.length
  }

  def appendBytes(bytes: Array[Byte], size: Int) {
    assert(bytes.length >= size)

    ensure(size)
    mb.storeBytes(offset, bytes, size)
    offset += size
  }

  def allocate(nBytes: Int): Int = {
    val currentOffset = offset
    ensure(nBytes)
    offset += nBytes
    currentOffset
  }

  def align(alignment: Int) {
    assert(alignment > 0, s"invalid alignment: $alignment")
    assert((alignment & (alignment - 1)) == 0, s"invalid alignment: $alignment") // power of 2
    offset = (offset + (alignment - 1)) & ~(alignment - 1)
  }

  def copyFrom(other: MemoryBuffer, readStart: Int, writeStart: Int, size: Int) {
    assert(writeStart <= (offset - size))
    mb.copyFrom(other.mb, readStart, writeStart, size)
  }

  def clear() {
    offset = 0
  }

  def copy(): MemoryBuffer = new MemoryBuffer(mb.copy((offset + 7) / 8))

  def result(): MemoryBlock = {
    val reqLength = (offset + 7) / 8
    val arr = new Array[Long](reqLength)
    Platform.copyMemory(mb.mem, Platform.LONG_ARRAY_OFFSET, arr, Platform.LONG_ARRAY_OFFSET, offset)
    new MemoryBlock(arr)
  }
}

case class RegionValue(region: MemoryBuffer, offset: Int)

class RegionValueBuilder(region: MemoryBuffer) {
  var start: Int = _

  val typestk = new mutable.Stack[Type]()
  val indexstk = new mutable.Stack[Int]()
  val offsetstk = new mutable.Stack[Int]()
  val elementsOffsetstk = new mutable.Stack[Int]()

  def current(): (Type, Int) = {
    val i = indexstk.head
    typestk.head match {
      case t: TStruct =>
        (t.fields(i).typ, offsetstk.head + t.byteOffsets(i))

      case t: TArray =>
        (t.elementType, elementsOffsetstk.head + i * UnsafeUtils.arrayElementSize(t.elementType))
    }
  }

  def start(top: TStruct) {
    assert(typestk.isEmpty && offsetstk.isEmpty && elementsOffsetstk.isEmpty && indexstk.isEmpty)

    region.align(top.alignment)
    val off = region.offset
    start = off
    region.allocate(top.byteSize)

    val nMissingBytes = (top.size + 7) / 8
    var i = 0
    while (i < nMissingBytes) {
      region.storeByte(off + i, 0)
      i += 1
    }

    typestk.push(top)
    offsetstk.push(off)
    indexstk.push(0)
  }

  def end(): Int = {
    val t = typestk.pop()
    offsetstk.pop()
    val last = indexstk.pop()
    assert(last == t.asInstanceOf[TStruct].size)

    assert(typestk.isEmpty && offsetstk.isEmpty && elementsOffsetstk.isEmpty && indexstk.isEmpty)

    start
  }

  def advance(): Unit = {
    indexstk.push(indexstk.pop + 1)
  }

  def startStruct(): Unit = {
    current() match {
      case (t: TStruct, off) =>
        val nMissingBytes = (t.size + 7) / 8
        var i = 0
        while (i < nMissingBytes) {
          region.storeByte(off + i, 0)
          i += 1
        }

        typestk.push(t)
        offsetstk.push(off)
        indexstk.push(0)
    }
  }

  def endStruct(): Unit = {
    typestk.head match {
      case t: TStruct =>
        typestk.pop()
        offsetstk.pop()
        val last = indexstk.pop()
        assert(last == t.size)

        advance()
    }
  }

  def startArray(length: Int): Unit = {
    current() match {
      case (t: TArray, off) =>
        region.align(4)
        val aoff = region.offset
        region.storeInt(off, aoff)

        val nMissingBytes = (length + 7) / 8
        region.allocate(4 + nMissingBytes)

        region.storeInt(aoff, length)

        var i = 0
        while (i < nMissingBytes) {
          region.storeByte(aoff + 4 + i, 0)
          i += 1
        }

        region.align(t.elementType.alignment)
        val elementsOff = region.offset

        region.allocate(length * UnsafeUtils.arrayElementSize(t.elementType))

        typestk.push(t)
        elementsOffsetstk.push(elementsOff)
        indexstk.push(0)
        offsetstk.push(aoff)
    }
  }

  def endArray(): Unit = {
    typestk.head match {
      case t: TArray =>
        typestk.pop()
        offsetstk.pop()
        elementsOffsetstk.pop()
        indexstk.pop()

        advance()
    }
  }

  def setMissing(): Unit = {
    val i = indexstk.head
    typestk.head match {
      case t: TStruct =>
        region.setBit(offsetstk.head, i)
      case t: TArray =>
        region.setBit(offsetstk.head + 4, i)
    }

    advance()
  }

  def addBoolean(b: Boolean): Unit =
    current() match {
      case (TBoolean, off) =>
        region.storeByte(off, b.toByte)
        advance()
    }

  def addInt(i: Int): Unit =
    current() match {
      case (TInt32, off) =>
        region.storeInt(off, i)
        advance()
    }

  def addLong(l: Long): Unit =
    current() match {
      case (TInt64, off) =>
        region.storeLong(off, l)
        advance()
    }

  def addFloat(f: Float): Unit =
    current() match {
      case (TFloat32, off) =>
        region.storeFloat(off, f)
        advance()
    }

  def addDouble(d: Double): Unit =
    current() match {
      case (TFloat64, off) =>
        region.storeDouble(off, d)
        advance()
    }

  def addBinary(bytes: Array[Byte]): Unit =
    current() match {
      case (TFloat64, off) =>
        region.align(4)
        val boff = region.offset
        region.storeInt(off, boff)

        region.appendInt(bytes.length)
        region.appendBytes(bytes)
        advance()
    }

  def addString(s: String): Unit =
    current() match {
      case (TString, off) =>
        region.align(4)
        val soff = region.offset
        region.storeInt(off, soff)

        val bytes = s.getBytes
        region.appendInt(bytes.length)
        region.appendBytes(bytes)
        advance()
    }

  def addRow(t: TStruct, r: Row) {
    assert(r != null)
    var i = 0
    while (i < t.size) {
      addAnnotation(t.fields(i).typ, r.get(i))
      i += 1
    }
  }

  def addAnnotation(t: Type, a: Annotation) {
    if (a == null)
      setMissing()
    else
      t match {
        case TBoolean => addBoolean(a.asInstanceOf[Boolean])
        case TInt32 => addInt(a.asInstanceOf[Int])
        case TInt64 => addLong(a.asInstanceOf[Long])
        case TFloat32 => addFloat(a.asInstanceOf[Float])
        case TFloat64 => addDouble(a.asInstanceOf[Double])
        case TString => addString(a.asInstanceOf[String])
        case TBinary => addBinary(a.asInstanceOf[Array[Byte]])
        case TArray(elementType) =>
          val is = a.asInstanceOf[IndexedSeq[Annotation]]
          startArray(is.length)
          var i = 0
          while (i < is.length) {
            addAnnotation(elementType, is(i))
            i += 1
          }
          endArray()
        case t: TStruct =>
          startStruct()
          addRow(t, a.asInstanceOf[Row])
          endStruct()

        case TSet(elementType) =>
          val s = a.asInstanceOf[Set[Annotation]]
          startArray(s.size)
          s.foreach { x => addAnnotation(elementType, x) }
          endArray()

        case TDict(keyType, valueType) =>
          val m = a.asInstanceOf[Map[Annotation, Annotation]]
          startArray(m.size)
          m.foreach { case (k, v) =>
            startStruct()
            addAnnotation(keyType, k)
            addAnnotation(valueType, v)
            endStruct()
          }
          endArray()

        case TVariant =>
          val v = a.asInstanceOf[Variant]
          startStruct()
          addString(v.contig)
          addInt(v.start)
          addString(v.ref)
          startArray(v.altAlleles.length)
          var i = 0
          while (i < v.altAlleles.length) {
            addAnnotation(TAltAllele, v.altAlleles(i))
            i += 1
          }
          endArray()
          endStruct()

        case TAltAllele =>
          val aa = a.asInstanceOf[AltAllele]
          startStruct()
          addString(aa.ref)
          addString(aa.alt)
          endStruct()

        case TCall =>
          addInt(a.asInstanceOf[Int])

        case TGenotype =>
          val g = a.asInstanceOf[Genotype]
          startStruct()

          val unboxedGT = g._unboxedGT
          if (unboxedGT >= 0)
            addInt(unboxedGT)
          else
            setMissing()

          val unboxedAD = g._unboxedAD
          if (unboxedAD == null)
            setMissing()
          else {
            startArray(unboxedAD.length)
            var i = 0
            while (i < unboxedAD.length) {
              addInt(unboxedAD(i))
              i += 1
            }
            endArray()
          }

          val unboxedDP = g._unboxedDP
          if (unboxedDP >= 0)
            addInt(unboxedDP)
          else
            setMissing()

          val unboxedGQ = g._unboxedGQ
          if (unboxedGQ >= 0)
            addInt(unboxedGQ)
          else
            setMissing()

          val unboxedPX = g._unboxedPX
          if (unboxedPX == null)
            setMissing()
          else {
            startArray(unboxedPX.length)
            var i = 0
            while (i < unboxedPX.length) {
              addInt(unboxedPX(i))
              i += 1
            }
            endArray()
          }

          addBoolean(g._fakeRef)
          addBoolean(g._isLinearScale)
          endStruct()

        case TLocus =>
          val l = a.asInstanceOf[Locus]
          startStruct()
          addString(l.contig)
          addInt(l.position)
          endStruct()

        case TInterval =>
          val i = a.asInstanceOf[Interval[Locus]]
          startStruct()
          addAnnotation(TLocus, i.start)
          addAnnotation(TLocus, i.end)
          endStruct()
      }
  }

  def result(): RegionValue = RegionValue(region, start)
}
