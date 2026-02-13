package is.hail.io

import is.hail.annotations.{Memory, Region}
import is.hail.io.compress.LZ4
import is.hail.utils.implicits.{toRichBoolean, ByteTrackingOutputStream}

import java.io._
import java.util.function.Supplier

import com.github.luben.zstd.{Zstd, ZstdCompressCtx}

trait OutputBuffer extends Closeable {
  def flush(): Unit

  override def close(): Unit

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

  def writeBoolean(b: Boolean): Unit =
    writeByte(b.toByte)
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

  override def indexOffset(): Long = out.asInstanceOf[ByteTrackingOutputStream].bytesWritten

  override def writeByte(b: Byte): Unit = out.write(b.toInt)

  override def writeInt(i: Int): Unit = {
    Memory.storeInt(buf, 0, i)
    out.write(buf, 0, 4)
  }

  override def writeLong(l: Long): Unit = {
    Memory.storeLong(buf, 0, l)
    out.write(buf, 0, 8)
  }

  override def writeFloat(f: Float): Unit = {
    Memory.storeFloat(buf, 0, f)
    out.write(buf, 0, 4)
  }

  override def writeDouble(d: Double): Unit = {
    Memory.storeDouble(buf, 0, d)
    out.write(buf, 0, 8)
  }

  override def writeBytes(region: Region, off: Long, n: Int): Unit =
    out.write(Region.loadBytes(off, n))

  override def writeBytes(addr: Long, n: Int): Unit =
    out.write(Region.loadBytes(addr, n))

  override def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit = {
    var i = 0
    while (i < n) {
      writeDouble(from(fromOff + i))
      i += 1
    }
  }
}

final class MemoryOutputBuffer(mb: MemoryBuffer) extends OutputBuffer {
  override def flush(): Unit = {}

  override def close(): Unit = {}

  override def indexOffset(): Long = ???

  override def writeByte(b: Byte): Unit = mb.writeByte(b)

  override def writeInt(i: Int): Unit = mb.writeInt(i)

  override def writeLong(l: Long): Unit = mb.writeLong(l)

  override def writeFloat(f: Float): Unit = mb.writeFloat(f)

  override def writeDouble(d: Double): Unit = mb.writeDouble(d)

  override def writeBytes(region: Region, off: Long, n: Int): Unit = mb.writeBytes(off, n)

  override def writeBytes(addr: Long, n: Int): Unit = mb.writeBytes(addr, n)

  override def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit = ???
}

final class LEB128OutputBuffer(out: OutputBuffer) extends OutputBuffer {
  override def flush(): Unit = out.flush()

  override def close(): Unit =
    out.close()

  override def indexOffset(): Long = out.indexOffset()

  override def writeByte(b: Byte): Unit = out.writeByte(b)

  override def writeInt(i: Int): Unit = {
    var j = i
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      out.writeByte(b.toByte)
    } while (j != 0)
  }

  override def writeLong(l: Long): Unit = {
    var j = l
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      out.writeByte(b.toByte)
    } while (j != 0)
  }

  override def writeFloat(f: Float): Unit = out.writeFloat(f)

  override def writeDouble(d: Double): Unit = out.writeDouble(d)

  override def writeBytes(region: Region, off: Long, n: Int): Unit = out.writeBytes(region, off, n)

  override def writeBytes(addr: Long, n: Int): Unit = out.writeBytes(addr, n)

  override def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit =
    out.writeDoubles(from, fromOff, n)
}

final class BlockingOutputBuffer(blockSize: Int, out: OutputBlockBuffer) extends OutputBuffer {
  private val buf: Array[Byte] = new Array[Byte](blockSize)
  private var off: Int = 0

  override def indexOffset(): Long = {
    if (off == blockSize)
      writeBlock()
    (out.getPos() << 16) | off
  }

  private def writeBlock(): Unit = {
    out.writeBlock(buf, off)
    off = 0
  }

  override def flush(): Unit = {
    writeBlock()
    out.flush()
  }

  override def close(): Unit = {
    flush()
    out.close()
  }

  override def writeByte(b: Byte): Unit = {
    if (off + 1 > buf.length)
      writeBlock()
    Memory.storeByte(buf, off.toLong, b)
    off += 1
  }

  override def writeInt(i: Int): Unit = {
    if (off + 4 > buf.length)
      writeBlock()
    Memory.storeInt(buf, off.toLong, i)
    off += 4
  }

  override def writeLong(l: Long): Unit = {
    if (off + 8 > buf.length)
      writeBlock()
    Memory.storeLong(buf, off.toLong, l)
    off += 8
  }

  override def writeFloat(f: Float): Unit = {
    if (off + 4 > buf.length)
      writeBlock()
    Memory.storeFloat(buf, off.toLong, f)
    off += 4
  }

  override def writeDouble(d: Double): Unit = {
    if (off + 8 > buf.length)
      writeBlock()
    Memory.storeDouble(buf, off.toLong, d)
    off += 8
  }

  override def writeBytes(fromRegion: Region, fromOff0: Long, n0: Int) = writeBytes(fromOff0, n0)

  override def writeBytes(addr0: Long, n0: Int): Unit = {
    assert(n0 >= 0)
    var addr = addr0
    var n = n0

    while (off + n > buf.length) {
      val p = buf.length - off
      Region.loadBytes(addr, buf, off.toLong, p.toLong)
      off += p
      addr += p
      n -= p
      assert(off == buf.length)
      writeBlock()
    }
    Region.loadBytes(addr, buf, off.toLong, n.toLong)
    off += n
  }

  override def writeDoubles(from: Array[Double], fromOff0: Int, n0: Int): Unit = {
    assert(n0 >= 0)
    assert(fromOff0 >= 0)
    assert(fromOff0 <= from.length - n0)
    var fromOff = fromOff0
    var n = n0

    while (off + (n << 3) > buf.length) {
      val p = (buf.length - off) >>> 3
      Memory.memcpy(buf, off.toLong, from, fromOff.toLong, p.toLong)
      off += (p << 3)
      fromOff += p
      n -= p
      writeBlock()
    }
    Memory.memcpy(buf, off.toLong, from, fromOff.toLong, n.toLong)
    off += (n << 3)
  }
}

final class StreamBlockOutputBuffer(out: OutputStream) extends OutputBlockBuffer {
  private val lenBuf = new Array[Byte](4)

  override def flush(): Unit =
    out.flush()

  override def close(): Unit =
    out.close()

  override def writeBlock(buf: Array[Byte], len: Int): Unit = {
    Memory.storeInt(lenBuf, 0, len)
    out.write(lenBuf, 0, 4)
    out.write(buf, 0, len)
  }

  override def getPos(): Long = out.asInstanceOf[ByteTrackingOutputStream].bytesWritten
}

final class LZ4OutputBlockBuffer(lz4: LZ4, blockSize: Int, out: OutputBlockBuffer)
    extends OutputBlockBuffer {
  private val comp = new Array[Byte](4 + lz4.maxCompressedLength(blockSize))

  override def flush(): Unit =
    out.flush()

  override def close(): Unit =
    out.close()

  override def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = lz4.compress(comp, 4, buf, decompLen)
    Memory.storeInt(comp, 0, decompLen) // decompLen
    out.writeBlock(comp, compLen + 4)
  }

  override def getPos(): Long = out.getPos()
}

final class LZ4SizeBasedCompressingOutputBlockBuffer(
  lz4: LZ4,
  blockSize: Int,
  minCompressionSize: Int,
  out: OutputBlockBuffer,
) extends OutputBlockBuffer {
  private val comp = new Array[Byte](8 + lz4.maxCompressedLength(blockSize))

  override def flush(): Unit =
    out.flush()

  override def close(): Unit =
    out.close()

  override def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
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

  override def getPos(): Long = out.getPos()
}

object ZstdCompressLib {
  val instance = ThreadLocal.withInitial(new Supplier[ZstdCompressCtx]() {
    override def get = {
      val zstd = new ZstdCompressCtx()
      zstd.setLevel(Zstd.defaultCompressionLevel())
      zstd.setChecksum(false)
      zstd
    }
  })
}

final class ZstdOutputBlockBuffer(blockSize: Int, out: OutputBlockBuffer)
    extends OutputBlockBuffer {
  private[this] val zstd = ZstdCompressLib.instance.get
  private[this] val comp = new Array[Byte](4 + Zstd.compressBound(blockSize.toLong).toInt)

  override def flush(): Unit = out.flush()

  override def close(): Unit = out.close()

  override def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = zstd.compressByteArray(comp, 4, comp.length - 4, buf, 0, decompLen)
    Memory.storeInt(comp, 0, decompLen.toInt)
    out.writeBlock(comp, compLen.toInt + 4)
  }

  override def getPos(): Long = out.getPos()
}

final class ZstdSizedBasedOutputBlockBuffer(
  blockSize: Int,
  minCompressionSize: Int,
  out: OutputBlockBuffer,
) extends OutputBlockBuffer {
  private[this] val zstd = ZstdCompressLib.instance.get
  private[this] val comp = new Array[Byte](4 + Zstd.compressBound(blockSize.toLong).toInt)

  override def flush(): Unit = out.flush()

  override def close(): Unit = out.close()

  override def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = if (decompLen < minCompressionSize) {
      System.arraycopy(buf, 0, comp, 4, decompLen)
      Memory.storeInt(comp, 0, 0)
      decompLen
    } else {
      val compLen = zstd.compressByteArray(comp, 4, comp.length - 4, buf, 0, decompLen)
      Memory.storeInt(comp, 0, (decompLen << 1) + 1)
      compLen.toInt
    }

    out.writeBlock(comp, compLen + 4)
  }

  override def getPos(): Long = out.getPos()
}
