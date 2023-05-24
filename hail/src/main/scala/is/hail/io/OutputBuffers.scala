package is.hail.io

import java.io._
import java.util

import is.hail.annotations.{Memory, Region}
import is.hail.io.compress.LZ4
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream

import com.github.luben.zstd.Zstd

trait OutputBuffer extends Closeable {
  def flush(): Unit

  def close(): Unit

  def indexOffset(): Long

  def writeByte(b: Byte): Unit

  def write(buf: Array[Byte]): Unit = write(buf, 0, buf.length)

  def write(buf: Array[Byte], startPos: Int, endPos: Int): Unit = {
    var i = startPos
    while (i < endPos) {
      writeByte(buf(i))
      i += 1
    }
  }

  def writeInt(i: Int): Unit

  def writeLong(l: Long): Unit

  def writeFloat(f: Float): Unit

  def writeDouble(d: Double): Unit

  def writeBytes(region: Region, off: Long, n: Int): Unit

  def writeBytes(addr: Long, n: Int): Unit

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit

  def writeDoubles(from: Array[Double]): Unit = writeDoubles(from, 0, from.length)

  def writeBoolean(b: Boolean) {
    writeByte(b.toByte)
  }

  def writeUTF(s: String): Unit = {
    val bytes = s.getBytes(utfCharset)
    writeInt(bytes.length)
    write(bytes)
  }
}

trait OutputBlockBuffer extends Spec with Closeable {
  def flush(): Unit

  def writeBlock(buf: Array[Byte], len: Int): Unit

  def getPos(): Long
}

final class StreamOutputBuffer(out: OutputStream) extends OutputBuffer {
  private val buf = new Array[Byte](8)

  override def flush(): Unit = out.flush()

  override def close(): Unit = out.close()

  def indexOffset(): Long = out.asInstanceOf[ByteTrackingOutputStream].bytesWritten

  override def writeByte(b: Byte): Unit = out.write(Array(b))

  override def writeInt(i: Int) {
    Memory.storeInt(buf, 0, i)
    out.write(buf, 0, 4)
  }

  def writeLong(l: Long) {
    Memory.storeLong(buf, 0, l)
    out.write(buf, 0, 8)
  }

  def writeFloat(f: Float) {
    Memory.storeFloat(buf, 0, f)
    out.write(buf, 0, 4)
  }

  def writeDouble(d: Double) {
    Memory.storeDouble(buf, 0, d)
    out.write(buf, 0, 8)
  }

  def writeBytes(region: Region, off: Long, n: Int): Unit =
    out.write(Region.loadBytes(off, n))

  def writeBytes(addr: Long, n: Int): Unit =
    out.write(Region.loadBytes(addr, n))

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int) {
    var i = 0
    while (i < n) {
      writeDouble(from(fromOff + i))
      i += 1
    }
  }
}

final class MemoryOutputBuffer(mb: MemoryBuffer) extends OutputBuffer {
  def flush() {}

  def close() {}

  def indexOffset(): Long = ???

  def writeByte(b: Byte): Unit = mb.writeByte(b)

  def writeInt(i: Int): Unit = mb.writeInt(i)

  def writeLong(l: Long): Unit = mb.writeLong(l)

  def writeFloat(f: Float): Unit = mb.writeFloat(f)

  def writeDouble(d: Double): Unit = mb.writeDouble(d)

  def writeBytes(region: Region, off: Long, n: Int): Unit = mb.writeBytes(off, n)

  def writeBytes(addr: Long, n: Int): Unit = mb.writeBytes(addr, n)

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit = ???
}

final class LEB128OutputBuffer(out: OutputBuffer) extends OutputBuffer {
  def flush(): Unit = out.flush()

  def close() {
    out.close()
  }

  def indexOffset(): Long = out.indexOffset()

  def writeByte(b: Byte): Unit = out.writeByte(b)

  def writeInt(i: Int): Unit = {
    var j = i
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      out.writeByte(b.toByte)
    } while (j != 0)
  }

  def writeLong(l: Long): Unit = {
    var j = l
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      out.writeByte(b.toByte)
    } while (j != 0)
  }

  def writeFloat(f: Float): Unit = out.writeFloat(f)

  def writeDouble(d: Double): Unit = out.writeDouble(d)

  def writeBytes(region: Region, off: Long, n: Int): Unit = out.writeBytes(region, off, n)

  def writeBytes(addr: Long, n: Int): Unit = out.writeBytes(addr, n)

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit = out.writeDoubles(from, fromOff, n)
}

final class BlockingOutputBuffer(blockSize: Int, out: OutputBlockBuffer) extends OutputBuffer {
  private val buf: Array[Byte] = new Array[Byte](blockSize)
  private var off: Int = 0

  def indexOffset(): Long = {
    if (off == blockSize)
      writeBlock()
    (out.getPos() << 16) | off
  }

  private def writeBlock() {
    out.writeBlock(buf, off)
    off = 0
  }

  def flush() {
    writeBlock()
    out.flush()
  }

  def close() {
    flush()
    out.close()
  }

  def writeByte(b: Byte) {
    if (off + 1 > buf.length)
      writeBlock()
    Memory.storeByte(buf, off, b)
    off += 1
  }

  def writeInt(i: Int) {
    if (off + 4 > buf.length)
      writeBlock()
    Memory.storeInt(buf, off, i)
    off += 4
  }

  def writeLong(l: Long) {
    if (off + 8 > buf.length)
      writeBlock()
    Memory.storeLong(buf, off, l)
    off += 8
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

  def writeBytes(fromRegion: Region, fromOff0: Long, n0: Int) = writeBytes(fromOff0, n0)

  def writeBytes(addr0: Long, n0: Int) {
    assert(n0 >= 0)
    var addr = addr0
    var n = n0

    while (off + n > buf.length) {
      val p = buf.length - off
      Region.loadBytes(addr, buf, off, p)
      off += p
      addr += p
      n -= p
      assert(off == buf.length)
      writeBlock()
    }
    Region.loadBytes(addr, buf, off, n)
    off += n
  }

  def writeDoubles(from: Array[Double], fromOff0: Int, n0: Int) {
    assert(n0 >= 0)
    assert(fromOff0 >= 0)
    assert(fromOff0 <= from.length - n0)
    var fromOff = fromOff0
    var n = n0

    while (off + (n << 3) > buf.length) {
      val p = (buf.length - off) >>> 3
      Memory.memcpy(buf, off, from, fromOff, p)
      off += (p << 3)
      fromOff += p
      n -= p
      writeBlock()
    }
    Memory.memcpy(buf, off, from, fromOff, n)
    off += (n << 3)
  }
}

final class StreamBlockOutputBuffer(out: OutputStream) extends OutputBlockBuffer {
  private val lenBuf = new Array[Byte](4)

  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeBlock(buf: Array[Byte], len: Int): Unit = {
    Memory.storeInt(lenBuf, 0, len)
    out.write(lenBuf, 0, 4)
    out.write(buf, 0, len)
  }

  def getPos(): Long = out.asInstanceOf[ByteTrackingOutputStream].bytesWritten
}

final class LZ4OutputBlockBuffer(lz4: LZ4, blockSize: Int, out: OutputBlockBuffer) extends OutputBlockBuffer {
  private val comp = new Array[Byte](4 + lz4.maxCompressedLength(blockSize))

  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = lz4.compress(comp, 4, buf, decompLen)
    Memory.storeInt(comp, 0, decompLen) // decompLen
    out.writeBlock(comp, compLen + 4)
  }

  def getPos(): Long = out.getPos()
}

final class LZ4SizeBasedCompressingOutputBlockBuffer(lz4: LZ4, blockSize: Int, minCompressionSize: Int, out: OutputBlockBuffer) extends OutputBlockBuffer {
  private val comp = new Array[Byte](8 + lz4.maxCompressedLength(blockSize))

  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    if (decompLen < minCompressionSize) {
      System.arraycopy(buf, 0, comp, 4, decompLen)
      Memory.storeInt(comp, 0, 0) // uncompressed
      out.writeBlock(comp, decompLen + 4)
    } else {
      val compLen = lz4.compress(comp, 8, buf, decompLen)
      Memory.storeInt(comp, 0, 1) // compressed
      Memory.storeInt(comp, 4, decompLen) // decompLen
      out.writeBlock(comp, compLen + 8)
    }
  }

  def getPos(): Long = out.getPos()
}

final class ZstdOutputBlockBuffer(blockSize: Int, out: OutputBlockBuffer) extends OutputBlockBuffer {
  private val comp = new Array[Byte](4 + Zstd.compressBound(blockSize).toInt)

  def flush(): Unit = out.flush()

  def close(): Unit = out.close()

  def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = Zstd.compressByteArray(comp, 4, comp.length - 4, buf, 0, decompLen, Zstd.defaultCompressionLevel())
    if (Zstd.isError(compLen))
      throw new com.github.luben.zstd.ZstdException(compLen)
    Memory.storeInt(comp, 0, decompLen.toInt)
    out.writeBlock(comp, compLen.toInt + 4)
  }

  def getPos(): Long = out.getPos()
}

final class ZstdSizedBasedOutputBlockBuffer(blockSize: Int, minCompressionSize: Int, out: OutputBlockBuffer) extends OutputBlockBuffer {
  private val comp = new Array[Byte](4 + Zstd.compressBound(blockSize).toInt)

  def flush(): Unit = out.flush()

  def close(): Unit = out.close()

  def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = if (decompLen < minCompressionSize) {
      System.arraycopy(buf, 0, comp, 4, decompLen)
      Memory.storeInt(comp, 0, 0)
      decompLen
    } else {
      val compLen = Zstd.compressByteArray(comp, 4, comp.length - 4, buf, 0, decompLen, Zstd.defaultCompressionLevel())
      if (Zstd.isError(compLen))
        throw new com.github.luben.zstd.ZstdException(compLen)
      Memory.storeInt(comp, 0, (decompLen << 1) + 1)
      compLen.toInt
    }

    out.writeBlock(comp, compLen + 4)
  }

  def getPos(): Long = out.getPos()
}
