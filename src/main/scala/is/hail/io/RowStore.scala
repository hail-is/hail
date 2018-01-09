package is.hail.io

import java.io.{InputStream, OutputStream}

import is.hail.annotations.{Memory, Region, RegionValue}
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.LZ4Utils
import org.apache.spark.rdd.RDD

class ArrayInputStream(var a: Array[Byte], var end: Int) extends InputStream {
  var off: Int = 0

  def this(a: Array[Byte]) = this(a, a.length)

  def clear() {
    off = 0
  }

  def read(): Int = {
    if (off == end)
      -1
    else {
      val c = a(off) & 0xff
      off += 1
      c
    }
  }

  override def read(to: Array[Byte], toOff: Int, len: Int): Int = {
    if (end == off)
      -1
    else {
      val p = math.min(len, end - off)
      System.arraycopy(a, off, to, toOff, p)
      off += p
      p
    }
  }

  override def skip(n: Long): Long = {
    assert(end != off)
    assert(n <= Int.MaxValue)
    val p = math.min(n.toInt, end - off)
    off += p
    p
  }

  override def available(): Int = end - off
}

class ArrayOutputStream(sizeHint: Int = 32) extends OutputStream {
  var a: Array[Byte] = new Array[Byte](sizeHint)
  var off: Int = 0

  private def ensure(n: Int) {
    val newEnd = off + n
    if (newEnd > a.length) {
      val newLength = math.max((a.length * 3) / 2, off + n)
      val newA = new Array[Byte](newLength)
      System.arraycopy(a, 0, newA, 0, off)
      a = newA
    }
  }

  def clear() {
    off = 0
  }

  def write(b: Int) {
    ensure(1)
    a(off) = b.toByte
    off += 1
  }

  override def write(from: Array[Byte], fromOff: Int, len: Int) {
    ensure(len)
    System.arraycopy(from, fromOff, a, off, len)
    off += len
  }
}

abstract class OutputBuffer {
  def flush(): Unit

  def writeByte(b: Byte): Unit

  def writeInt(i: Int): Unit

  def writeLong(l: Long): Unit

  def writeFloat(f: Float): Unit

  def writeDouble(d: Double): Unit

  def writeBytes(region: Region, off: Long, n: Int): Unit

  def writeBoolean(b: Boolean) {
    writeByte(b.toByte)
  }
}

object LZ4Buffer {
  final val blockSize: Int = 32 * 1024
}

class LZ4OutputBuffer(out: OutputStream) extends OutputBuffer {
  val buf: Array[Byte] = new Array[Byte](LZ4Buffer.blockSize)
  var off: Int = 0

  val comp = new Array[Byte](8 + LZ4Utils.maxCompressedLength(LZ4Buffer.blockSize))

  private def writeBlock() {
    if (off > 0) {
      val compLen = LZ4Utils.compress(comp, 8, buf, off)
      Memory.storeInt(comp, 0, compLen)
      Memory.storeInt(comp, 4, off) // decompLen

      out.write(comp, 0, 8 + compLen)

      off = 0
    }
  }

  def flush() {
    writeBlock()
  }

  def writeByte(b: Byte) {
    if (off + 1 > buf.length)
      writeBlock()
    Memory.storeByte(buf, off, b)
    off += 1
  }

  def writeInt(i: Int) {
    if (off + 5 > buf.length)
      writeBlock()

    var j = i
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      Memory.storeByte(buf, off, b.toByte)
      off += 1
    } while (j != 0)
  }

  def writeLong(l: Long) {
    if (off + 10 > buf.length)
      writeBlock()

    var j = l
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      Memory.storeByte(buf, off, b.toByte)
      off += 1
    } while (j != 0)
  }

  def writeFloat(f: Float) {
    if (off + 4 > buf.length)
      writeBlock()
    Memory.storeFloat(buf, off, f)
    off += 4
  }

  def writeDouble(d: Double) {
    if (off + 8 > buf.length)
      writeBlock()
    Memory.storeDouble(buf, off, d)
    off += 8
  }

  def writeBytes(fromRegion: Region, fromOff0: Long, n0: Int) {
    assert(n0 >= 0)
    var fromOff = fromOff0
    var n = n0

    while (off + n > buf.length) {
      val p = buf.length - off
      fromRegion.loadBytes(fromOff, buf, off, p)
      off += p
      fromOff += p
      n -= p
      assert(off == buf.length)
      writeBlock()
    }
    fromRegion.loadBytes(fromOff, buf, off, n)
    off += n
  }
}

abstract class InputBuffer {
  def readByte(): Byte

  def readInt(): Int

  def readLong(): Long

  def readFloat(): Float

  def readDouble(): Double

  def readBytes(toRegion: Region, toOff: Long, n: Int)

  def readBoolean(): Boolean = readByte() != 0
}

class LZ4InputBuffer(in: InputStream) extends InputBuffer {
  private val buf = new Array[Byte](LZ4Buffer.blockSize)
  private var end: Int = 0
  private var off: Int = 0

  private val comp = new Array[Byte](8 + LZ4Utils.maxCompressedLength(LZ4Buffer.blockSize))

  private def readBlock() {
    assert(off == end)

    // read the header
    in.readFully(comp, 0, 4)
    val compLen = Memory.loadInt(comp, 0)

    in.readFully(comp, 4, 4 + compLen)
    val decompLen = Memory.loadInt(comp, 4)

    LZ4Utils.decompress(buf, 0, decompLen, comp, 8, compLen)

    off = 0
    end = decompLen
  }

  private def ensure(n: Int) {
    if (off == end)
      readBlock()
    assert(off + n <= end)
  }

  def readByte(): Byte = {
    ensure(1)
    val b = Memory.loadByte(buf, off)
    off += 1
    b
  }

  def readInt(): Int = {
    ensure(1)

    var b: Byte = Memory.loadByte(buf, off)
    off += 1
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = Memory.loadByte(buf, off)
      off += 1
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    x
  }

  def readLong(): Long = {
    ensure(1)

    var b: Byte = Memory.loadByte(buf, off)
    off += 1
    var x: Long = b & 0x7fL
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = Memory.loadByte(buf, off)
      off += 1
      x |= ((b & 0x7fL) << shift)
      shift += 7
    }

    x
  }

  def readFloat(): Float = {
    ensure(4)
    val f = Memory.loadFloat(buf, off)
    off += 4
    f
  }

  def readDouble(): Double = {
    ensure(8)
    val d = Memory.loadDouble(buf, off)
    off += 8
    d
  }

  def readBytes(toRegion: Region, toOff0: Long, n0: Int) {
    assert(n0 >= 0)
    var toOff = toOff0
    var n = n0

    while (n > 0) {
      if (end == off)
        readBlock()
      val p = math.min(end - off, n)
      assert(p > 0)
      toRegion.storeBytes(toOff, buf, off, p)
      toOff += p
      n -= p
      off += p
    }
  }
}

final class Decoder(in: InputBuffer) {
  def readByte(): Byte = in.readByte()

  def readBinary(region: Region, off: Long) {
    val length = in.readInt()
    val boff = region.allocate(4, 4 + length)
    region.storeAddress(off, boff)
    region.storeInt(boff, length)
    in.readBytes(region, boff + 4, length)
  }

  def readArray(t: TArray, region: Region): Long = {
    val length = in.readInt()

    val contentSize = t.contentsByteSize(length)
    val aoff = region.allocate(t.contentsAlignment, contentSize)

    region.storeInt(aoff, length)
    if (!t.elementType.required) {
      val nMissingBytes = (length + 7) >>> 3
      in.readBytes(region, aoff + 4, nMissingBytes)
    }

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = t.elementByteSize

    if (t.elementType.isInstanceOf[TInt32]) { // fast path
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          region.storeInt(off, in.readInt())
        }
        i += 1
      }
    } else {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          t.elementType match {
            case t2: TStruct => readStruct(t2, region, off)
            case t2: TArray =>
              val aoff = readArray(t2, region)
              region.storeAddress(off, aoff)
            case _: TBoolean => region.storeByte(off, in.readBoolean().toByte)
            case _: TInt64 => region.storeLong(off, in.readLong())
            case _: TFloat32 => region.storeFloat(off, in.readFloat())
            case _: TFloat64 => region.storeDouble(off, in.readDouble())
            case _: TBinary => readBinary(region, off)
          }
        }
        i += 1
      }
    }

    aoff
  }

  def readStruct(t: TStruct, region: Region, offset: Long) {
    val nMissingBytes = t.nMissingBytes
    in.readBytes(region, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(region, offset, i)) {
        val f = t.fields(i)
        val off = offset + t.byteOffsets(i)
        f.typ match {
          case t2: TStruct => readStruct(t2, region, off)
          case t2: TArray =>
            val aoff = readArray(t2, region)
            region.storeAddress(off, aoff)
          case _: TBoolean => region.storeByte(off, in.readBoolean().toByte)
          case _: TInt32 => region.storeInt(off, in.readInt())
          case _: TInt64 => region.storeLong(off, in.readLong())
          case _: TFloat32 => region.storeFloat(off, in.readFloat())
          case _: TFloat64 => region.storeDouble(off, in.readDouble())
          case _: TBinary => readBinary(region, off)
        }
      }
      i += 1
    }
  }

  def readRegionValue(t: Type, region: Region): Long = {
    val f = t.fundamentalType
    f match {
      case t: TStruct =>
        val start = region.allocate(t.alignment, t.byteSize)
        readStruct(t, region, start)
        start

      case t: TArray =>
        readArray(t, region)
    }
  }
}

final class Encoder(out: OutputBuffer) {
  def flush() { out.flush() }

  def writeByte(b: Byte): Unit = out.writeByte(b)

  def writeBinary(region: Region, offset: Long) {
    val boff = region.loadAddress(offset)
    val length = region.loadInt(boff)
    out.writeInt(length)
    out.writeBytes(region, boff + 4, length)
  }

  def writeArray(t: TArray, region: Region, aoff: Long) {
    val length = region.loadInt(aoff)

    out.writeInt(length)
    if (!t.elementType.required) {
      val nMissingBytes = (length + 7) >>> 3
      out.writeBytes(region, aoff + 4, nMissingBytes)
    }

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = t.elementByteSize
    if (t.elementType.isInstanceOf[TInt32]) { // fast case
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          out.writeInt(region.loadInt(off))
        }
        i += 1
      }
    } else {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          t.elementType match {
            case t2: TStruct => writeStruct(t2, region, off)
            case t2: TArray => writeArray(t2, region, region.loadAddress(off))
            case _: TBoolean => out.writeBoolean(region.loadByte(off) != 0)
            case _: TInt64 => out.writeLong(region.loadLong(off))
            case _: TFloat32 => out.writeFloat(region.loadFloat(off))
            case _: TFloat64 => out.writeDouble(region.loadDouble(off))
            case _: TBinary => writeBinary(region, off)
          }
        }

        i += 1
      }
    }
  }

  def writeStruct(t: TStruct, region: Region, offset: Long) {
    val nMissingBytes = t.nMissingBytes
    out.writeBytes(region, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(region, offset, i)) {
        val off = offset + t.byteOffsets(i)
        t.fields(i).typ match {
          case t2: TStruct => writeStruct(t2, region, off)
          case t2: TArray => writeArray(t2, region, region.loadAddress(off))
          case _: TBoolean => out.writeBoolean(region.loadByte(off) != 0)
          case _: TInt32 => out.writeInt(region.loadInt(off))
          case _: TInt64 => out.writeLong(region.loadLong(off))
          case _: TFloat32 => out.writeFloat(region.loadFloat(off))
          case _: TFloat64 => out.writeDouble(region.loadDouble(off))
          case _: TBinary => writeBinary(region, off)
        }
      }

      i += 1
    }
  }

  def writeRegionValue(t: Type, region: Region, offset: Long) {
    val f = t.fundamentalType
    (f: @unchecked) match {
      case t: TStruct =>
        writeStruct(t, region, offset)
      case t: TArray =>
        writeArray(t, region, offset)
    }
  }
}

object RichRDDRegionValue {
  def writeRowsPartition(t: TStruct)(i: Int, it: Iterator[RegionValue], os: OutputStream): Long = {
    val en = new Encoder(new LZ4OutputBuffer(os))
    var rowCount = 0L
    
    it.foreach { rv =>
      en.writeByte(1)
      en.writeRegionValue(t, rv.region, rv.offset)
      rowCount += 1
    }

    en.writeByte(0) // end
    en.flush()
    os.close()
    
    rowCount
  }
}

class RichRDDRegionValue(val rdd: RDD[RegionValue]) extends AnyVal {
  def writeRows(path: String, t: TStruct): Array[Long] = {
    rdd.writePartitions(path, RichRDDRegionValue.writeRowsPartition(t))
  }
}