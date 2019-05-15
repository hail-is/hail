package is.hail.io

import java.io._
import java.util

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitUtils, EstimableEmitter, MethodBuilderLike}
import is.hail.expr.types.MatrixType
import is.hail.expr.types.physical._
import is.hail.io.compress.LZ4Utils
import is.hail.nativecode._
import is.hail.rvd.{AbstractRVDSpec, OrderedRVDSpec, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.{HailContext, cxx}
import is.hail.io.fs.FS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{ExposedMetrics, TaskContext}
import org.json4s.jackson.JsonMethods
import org.json4s.{Extraction, JValue}

trait BufferSpec extends Serializable {
  def buildInputBuffer(in: InputStream): InputBuffer

  def buildOutputBuffer(out: OutputStream): OutputBuffer

  def nativeOutputBufferType: String

  def nativeInputBufferType: String
}

final case class LEB128BufferSpec(child: BufferSpec) extends BufferSpec {
  def buildInputBuffer(in: InputStream): InputBuffer = new LEB128InputBuffer(child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new LEB128OutputBuffer(child.buildOutputBuffer(out))

  def nativeOutputBufferType: String = s"LEB128OutputBuffer<${ child.nativeOutputBufferType }>"

  def nativeInputBufferType: String = s"LEB128InputBuffer<${ child.nativeInputBufferType }>"
}

final case class BlockingBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BufferSpec {
  def buildInputBuffer(in: InputStream): InputBuffer = new BlockingInputBuffer(blockSize, child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new BlockingOutputBuffer(blockSize, child.buildOutputBuffer(out))

  def nativeOutputBufferType: String = s"BlockingOutputBuffer<$blockSize, ${ child.nativeOutputBufferType }>"

  def nativeInputBufferType: String = s"BlockingInputBuffer<$blockSize, ${ child.nativeInputBufferType }>"
}

trait BlockBufferSpec extends Serializable {
  def buildInputBuffer(in: InputStream): InputBlockBuffer

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer

  def nativeOutputBufferType: String

  def nativeInputBufferType: String
}

final case class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BlockBufferSpec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer = new LZ4InputBlockBuffer(blockSize, child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer = new LZ4OutputBlockBuffer(blockSize, child.buildOutputBuffer(out))

  def nativeOutputBufferType: String = s"LZ4OutputBlockBuffer<${ 4 + LZ4Utils.maxCompressedLength(blockSize) }, ${ child.nativeOutputBufferType }>"

  def nativeInputBufferType: String = s"LZ4InputBlockBuffer<${ 4 + LZ4Utils.maxCompressedLength(blockSize) }, ${ child.nativeInputBufferType }>"
}

object StreamBlockBufferSpec {
  def extract(jv: JValue): StreamBlockBufferSpec = new StreamBlockBufferSpec
}

final class StreamBlockBufferSpec extends BlockBufferSpec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer = new StreamBlockInputBuffer(in)

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer = new StreamBlockOutputBuffer(out)

  def nativeOutputBufferType: String = s"StreamOutputBlockBuffer"

  def nativeInputBufferType: String = s"StreamInputBlockBuffer"

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBlockBufferSpec]
}

final class StreamBufferSpec extends BufferSpec {
  override def buildInputBuffer(in: InputStream): InputBuffer = new StreamInputBuffer(in)

  override def buildOutputBuffer(out: OutputStream): OutputBuffer = new StreamOutputBuffer(out)

  override def nativeInputBufferType: String = s"StreamInputBuffer"

  override def nativeOutputBufferType: String = s"StreamOutputBuffer"

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBufferSpec]
}

object CodecSpec {
  val default: CodecSpec = new PackCodecSpec(
    new LEB128BufferSpec(
      new BlockingBufferSpec(32 * 1024,
        new LZ4BlockBufferSpec(32 * 1024,
          new StreamBlockBufferSpec))))

  val defaultUncompressed = new PackCodecSpec(
    new BlockingBufferSpec(32 * 1024,
      new StreamBlockBufferSpec))

  val unblockedUncompressed = new PackCodecSpec(new StreamBufferSpec)

  val baseBufferSpecs: Array[BufferSpec] = Array(
    new BlockingBufferSpec(64 * 1024,
      new StreamBlockBufferSpec),
    new BlockingBufferSpec(32 * 1024,
      new LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)),
    new StreamBufferSpec)

  val bufferSpecs: Array[BufferSpec] = baseBufferSpecs.flatMap { blockSpec =>
    Array(blockSpec, new LEB128BufferSpec(blockSpec))
  }

  val codecSpecs: Array[CodecSpec] = bufferSpecs.flatMap { bufferSpec =>
    Array(new PackCodecSpec(bufferSpec))
  }

  val supportedCodecSpecs: Array[CodecSpec] = bufferSpecs.flatMap { bufferSpec =>
    Array(new PackCodecSpec(bufferSpec))
  }
}

trait CodecSpec extends Serializable {
  def buildEncoder(t: PType, requestedType: PType): (OutputStream) => Encoder

  def buildEncoder(t: PType): (OutputStream) => Encoder = buildEncoder(t, t)

  def buildDecoder(t: PType, requestedType: PType): (InputStream) => Decoder

  def buildNativeDecoderClass(t: PType, requestedType: PType, tub: cxx.TranslationUnitBuilder): cxx.Class

  def buildNativeEncoderClass(t: PType, tub: cxx.TranslationUnitBuilder): cxx.Class

  // FIXME: is there a better place for this to live?
  def decodeRDD(t: PType, bytes: RDD[Array[Byte]]): ContextRDD[RVDContext, RegionValue] = {
    val dec = buildDecoder(t, t)
    ContextRDD.weaken[RVDContext](bytes).cmapPartitions { (ctx, it) =>
      val rv = RegionValue(ctx.region)
      it.map(RegionValue.fromBytes(dec, ctx.region, rv))
    }
  }

  override def toString: String = {
    implicit val formats = AbstractRVDSpec.formats
    val jv = Extraction.decompose(this)
    JsonMethods.compact(JsonMethods.render(jv))
  }
}

object ShowBuf {

  def apply(buf: Array[Byte], pos: Int, n: Int): Unit = {
    val sb = new StringBuilder()
    val len = if (n < 32) n else 32
    var j = 0
    while (j < len) {
      val x = (buf(pos + j).toInt & 0xff)
      if (x <= 0xf) sb.append(s" 0${ x.toHexString }") else sb.append(s" ${ x.toHexString }")
      if ((j & 0x7) == 0x7) sb.append("\n")
      j += 1
    }
    System.err.println(sb.toString())
  }

  def apply(addr: Long, n: Int): Unit = {
    val sb = new StringBuilder()
    val len = if (n < 32) n else 32
    var j = 0
    while (j < len) {
      val x = (Memory.loadByte(addr + j).toInt & 0xff)
      if (x <= 0xf) sb.append(s" 0${ x.toHexString }") else sb.append(s" ${ x.toHexString }")
      if ((j & 0x7) == 0x7) sb.append("\n")
      j += 1
    }
    System.err.println(sb.toString())
  }

}

final case class PackCodecSpec(child: BufferSpec) extends CodecSpec {

  def buildEncoder(t: PType, requestedType: PType): (OutputStream) => Encoder = {
    if (HailContext.isInitialized && HailContext.get.flags.get("cpp") != null && requestedType == t) {
      val e: NativeEncoderModule = cxx.PackEncoder.buildModule(t, child)
      (out: OutputStream) => new NativePackEncoder(out, e)
    } else {
      val f = EmitPackEncoder(t, requestedType)
      out: OutputStream => new CompiledPackEncoder(child.buildOutputBuffer(out), f)
    }
  }

  def buildDecoder(t: PType, requestedType: PType): (InputStream) => Decoder = {
    if (HailContext.isInitialized && HailContext.get.flags != null && HailContext.get.flags.get("cpp") != null) {
      val d: NativeDecoderModule = cxx.PackDecoder.buildModule(t, requestedType, child)
      (in: InputStream) => new NativePackDecoder(in, d)
    } else {
      val f = EmitPackDecoder(t, requestedType)
      (in: InputStream) => new CompiledPackDecoder(child.buildInputBuffer(in), f)
    }
  }

  def buildNativeDecoderClass(t: PType, requestedType: PType, tub: cxx.TranslationUnitBuilder): cxx.Class = cxx.PackDecoder(t, requestedType, child, tub)

  def buildNativeEncoderClass(t: PType, tub: cxx.TranslationUnitBuilder): cxx.Class = cxx.PackEncoder(t, child, tub)
}

trait OutputBlockBuffer extends Closeable {
  def writeBlock(buf: Array[Byte], len: Int): Unit
}

trait InputBlockBuffer extends Closeable {
  def close(): Unit

  def readBlock(buf: Array[Byte]): Int
}

final class StreamBlockOutputBuffer(out: OutputStream) extends OutputBlockBuffer {
  private val lenBuf = new Array[Byte](4)

  def close() {
    out.close()
  }

  def writeBlock(buf: Array[Byte], len: Int): Unit = {
    Memory.storeInt(lenBuf, 0, len)
    out.write(lenBuf, 0, 4)
    out.write(buf, 0, len)
  }
}

final class StreamBlockInputBuffer(in: InputStream) extends InputBlockBuffer {
  private val lenBuf = new Array[Byte](4)

  def close() {
    in.close()
  }

  def readBlock(buf: Array[Byte]): Int = {
    in.readFully(lenBuf, 0, 4)
    val len = Memory.loadInt(lenBuf, 0)
    assert(len >= 0)
    assert(len <= buf.length)
    in.readFully(buf, 0, len)
    len
  }
}

final class MemoryBuffer extends Serializable {
  var mem: Array[Byte] = new Array[Byte](8)
  var pos: Int = 0
  var end: Int = 0

  def capacity: Int = mem.length

  def clear() {
    pos = 0
    end = 0
  }

  def clearPos() {
    pos = 0
  }

  def grow(n: Int) {
    mem = util.Arrays.copyOf(mem, math.max(capacity * 2, end + n))
  }

  def copyFrom(src: MemoryBuffer) {
    mem = util.Arrays.copyOf(src.mem, src.capacity)
    end = src.end
    pos = src.pos
  }

  def writeByte(b: Byte) {
    if (end + 1 > capacity)
      grow(1)
    Memory.storeByte(mem, end, b)
    end += 1
  }

  def writeInt(i: Int) {
    if (end + 4 > capacity)
      grow(4)
    Memory.storeInt(mem, end, i)
    end += 4
  }

  def writeLong(i: Long) {
    if (end + 8 > capacity)
      grow(8)
    Memory.storeLong(mem, end, i)
    end += 8
  }

  def writeFloat(i: Float) {
    if (end + 4 > capacity)
      grow(4)
    Memory.storeFloat(mem, end, i)
    end += 4
  }

  def writeDouble(i: Double) {
    if (end + 8 > capacity)
      grow(8)
    Memory.storeDouble(mem, end, i)
    end += 8
  }

  def writeBytes(region: Region, off: Long, n: Int) {
    if (end + n > capacity)
      grow(n)
    Memory.memcpy(mem, end, off, n)
    end += n
  }

  def readByte(): Byte = {
    assert(pos + 1 <= end)
    val b = Memory.loadByte(mem, pos)
    pos += 1
    b
  }

  def readInt(): Int = {
    assert(pos + 4 <= end)
    val i = Memory.loadInt(mem, pos)
    pos += 4
    i
  }

  def readLong(): Long = {
    assert(pos + 8 <= end)
    val l = Memory.loadLong(mem, pos)
    pos += 8
    l
  }

  def readFloat(): Float = {
    assert(pos + 4 <= end)
    val f = Memory.loadFloat(mem, pos)
    pos += 4
    f
  }

  def readDouble(): Double = {
    assert(pos + 8 <= end)
    val d = Memory.loadDouble(mem, pos)
    pos += 8
    d
  }

  def readBytes(toRegion: Region, toOff: Long, n: Int) {
    assert(pos + n <= end)
    Memory.memcpy(toOff, mem, pos, n)
    pos += n
  }

  def skipByte() {
    assert(pos + 1 <= end)
    pos += 1
  }

  def skipInt() {
    assert(pos + 4 <= end)
    pos += 4
  }

  def skipLong() {
    assert(pos + 8 <= end)
    pos += 8
  }

  def skipFloat() {
    assert(pos + 4 <= end)
    pos += 4
  }

  def skipDouble() {
    assert(pos + 8 <= end)
    pos += 8
  }

  def skipBytes(n: Int) {
    assert(pos + n <= end)
    pos += n
  }
}

final class MemoryInputBuffer(mb: MemoryBuffer) extends InputBuffer {
  def close() {}

  def readByte(): Byte = mb.readByte()

  def readInt(): Int = mb.readInt()

  def readLong(): Long = mb.readLong()

  def readFloat(): Float = mb.readFloat()

  def readDouble(): Double = mb.readDouble()

  def readBytes(toRegion: Region, toOff: Long, n: Int): Unit = mb.readBytes(toRegion, toOff, n)

  def skipByte(): Unit = mb.skipByte()

  def skipInt(): Unit = mb.skipInt()

  def skipLong(): Unit = mb.skipLong()

  def skipFloat(): Unit = mb.skipFloat()

  def skipDouble(): Unit = mb.skipDouble()

  def skipBytes(n: Int): Unit = mb.skipBytes(n)

  def readDoubles(to: Array[Double], off: Int, n: Int): Unit = ???
}

final class MemoryOutputBuffer(mb: MemoryBuffer) extends OutputBuffer {
  def flush() {}

  def close() {}

  def writeByte(b: Byte): Unit = mb.writeByte(b)

  def writeInt(i: Int): Unit = mb.writeInt(i)

  def writeLong(l: Long): Unit = mb.writeLong(l)

  def writeFloat(f: Float): Unit = mb.writeFloat(f)

  def writeDouble(d: Double): Unit = mb.writeDouble(d)

  def writeBytes(region: Region, off: Long, n: Int): Unit = mb.writeBytes(region, off, n)

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit = ???
}

trait OutputBuffer extends Closeable {
  def flush(): Unit

  def close(): Unit

  def writeByte(b: Byte): Unit

  def writeInt(i: Int): Unit

  def writeLong(l: Long): Unit

  def writeFloat(f: Float): Unit

  def writeDouble(d: Double): Unit

  def writeBytes(region: Region, off: Long, n: Int): Unit

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit

  def writeDoubles(from: Array[Double]): Unit = writeDoubles(from, 0, from.length)

  def writeBoolean(b: Boolean) {
    writeByte(b.toByte)
  }
}

final class StreamOutputBuffer(out: OutputStream) extends OutputBuffer {
  private val buf = new Array[Byte](8)

  override def flush(): Unit = out.flush()

  override def close(): Unit = out.close()

  override def writeByte(b: Byte): Unit = out.write(Array(b))

  override def writeInt(i: Int) {
    Memory.storeInt(buf, 0, i)
    out.write(buf, 0, 4)
  }

  def writeLong(l: Long) {
    Memory.storeLong(buf, 0, l)
    out.write(buf)
  }

  def writeFloat(f: Float) {
    Memory.storeFloat(buf, 0, f)
    out.write(buf, 0, 4)
  }

  def writeDouble(d: Double) {
    Memory.storeDouble(buf, 0, d)
    out.write(buf)
  }

  def writeBytes(region: Region, off: Long, n: Int): Unit = out.write(region.loadBytes(off, n))

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int) {
    var i = 0
    while (i < n) {
      writeDouble(from(fromOff + i))
      i += 1
    }
  }
}

final class LEB128OutputBuffer(out: OutputBuffer) extends OutputBuffer {
  def flush(): Unit = out.flush()

  def close() {
    out.close()
  }

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

  def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit = out.writeDoubles(from, fromOff, n)
}

final class LZ4OutputBlockBuffer(blockSize: Int, out: OutputBlockBuffer) extends OutputBlockBuffer {
  private val comp = new Array[Byte](4 + LZ4Utils.maxCompressedLength(blockSize))

  def close() {
    out.close()
  }

  def writeBlock(buf: Array[Byte], decompLen: Int): Unit = {
    val compLen = LZ4Utils.compress(comp, 4, buf, decompLen)
    Memory.storeInt(comp, 0, decompLen) // decompLen
    out.writeBlock(comp, compLen + 4)
  }
}

final class BlockingOutputBuffer(blockSize: Int, out: OutputBlockBuffer) extends OutputBuffer {
  private val buf: Array[Byte] = new Array[Byte](blockSize)
  private var off: Int = 0

  private def writeBlock() {
    out.writeBlock(buf, off)
    off = 0
  }

  def flush() {
    writeBlock()
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

trait InputBuffer extends Closeable {
  def close(): Unit

  def readByte(): Byte

  def readInt(): Int

  def readLong(): Long

  def readFloat(): Float

  def readDouble(): Double

  def readBytes(toRegion: Region, toOff: Long, n: Int): Unit

  def skipBoolean(): Unit = skipByte()

  def skipByte(): Unit

  def skipInt(): Unit

  def skipLong(): Unit

  def skipFloat(): Unit

  def skipDouble(): Unit

  def skipBytes(n: Int): Unit

  def readDoubles(to: Array[Double], off: Int, n: Int): Unit

  def readDoubles(to: Array[Double]): Unit = readDoubles(to, 0, to.length)

  def readBoolean(): Boolean = readByte() != 0
}

final class StreamInputBuffer(in: InputStream) extends InputBuffer {
  private val buff = new Array[Byte](8)

  def close(): Unit = in.close()

  def readByte(): Byte = {
    in.read(buff, 0, 1)
    Memory.loadByte(buff, 0)
  }

  def readInt(): Int = {
    in.read(buff, 0, 4)
    Memory.loadInt(buff, 0)
  }

  def readLong(): Long = {
    in.read(buff)
    Memory.loadLong(buff, 0)
  }

  def readFloat(): Float = {
    in.read(buff, 0, 4)
    Memory.loadFloat(buff, 0)
  }

  def readDouble(): Double = {
    in.read(buff)
    Memory.loadDouble(buff, 0)
  }

  def readBytes(toRegion: Region, toOff: Long, n: Int): Unit = {
    toRegion.storeBytes(toOff, Array.tabulate(n)(_ => readByte()))
  }

  def skipByte(): Unit = in.skip(1)

  def skipInt(): Unit = in.skip(4)

  def skipLong(): Unit = in.skip(8)

  def skipFloat(): Unit = in.skip(4)

  def skipDouble(): Unit = in.skip(8)

  def skipBytes(n: Int): Unit = in.skip(n)

  def readDoubles(to: Array[Double], off: Int, n: Int): Unit = {
    var i = 0
    while (i < n) {
      to(off + i) = readDouble()
      i += 1
    }
  }
}

final class LEB128InputBuffer(in: InputBuffer) extends InputBuffer {
  def close() {
    in.close()
  }

  def readByte(): Byte = {
    in.readByte()
  }

  def readInt(): Int = {
    var b: Byte = readByte()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = readByte()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }
    x
  }

  def readLong(): Long = {
    var b: Byte = readByte()
    var x: Long = b & 0x7fL
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = readByte()
      x |= ((b & 0x7fL) << shift)
      shift += 7
    }
    x
  }

  def readFloat(): Float = in.readFloat()

  def readDouble(): Double = in.readDouble()

  def readBytes(toRegion: Region, toOff: Long, n: Int): Unit = in.readBytes(toRegion, toOff, n)

  def skipByte(): Unit = in.skipByte()

  def skipInt() {
    var b: Byte = readByte()
    while ((b & 0x80) != 0)
      b = readByte()
  }

  def skipLong() {
    var b: Byte = readByte()
    while ((b & 0x80) != 0)
      b = readByte()
  }

  def skipFloat(): Unit = in.skipFloat()

  def skipDouble(): Unit = in.skipDouble()

  def skipBytes(n: Int): Unit = in.skipBytes(n)

  def readDoubles(to: Array[Double], toOff: Int, n: Int): Unit = in.readDoubles(to, toOff, n)
}

final class LZ4InputBlockBuffer(blockSize: Int, in: InputBlockBuffer) extends InputBlockBuffer {
  private val comp = new Array[Byte](4 + LZ4Utils.maxCompressedLength(blockSize))
  private var decompBuf = new Array[Byte](blockSize)
  private var pos = 0
  private var lim = 0

  def close() {
    in.close()
  }

  def readBlock(buf: Array[Byte]): Int = {
    val blockLen = in.readBlock(comp)
    val result = if (blockLen == -1) {
      -1
    } else {
      val compLen = blockLen - 4
      val decompLen = Memory.loadInt(comp, 0)
      LZ4Utils.decompress(buf, 0, decompLen, comp, 4, compLen)
      decompLen
    }
    lim = result
    result
  }
}

final class BlockingInputBuffer(blockSize: Int, in: InputBlockBuffer) extends InputBuffer {
  private val buf = new Array[Byte](blockSize)
  private var end: Int = 0
  private var off: Int = 0

  private def readBlock() {
    assert(off == end)
    end = in.readBlock(buf)
    off = 0
  }

  private def ensure(n: Int) {
    if (off == end)
      readBlock()
    assert(off + n <= end)
  }

  def close() {
    in.close()
  }

  def readByte(): Byte = {
    ensure(1)
    val b = Memory.loadByte(buf, off)
    off += 1
    b
  }

  def readInt(): Int = {
    ensure(4)
    val i = Memory.loadInt(buf, off)
    off += 4
    i
  }

  def readLong(): Long = {
    ensure(8)
    val l = Memory.loadLong(buf, off)
    off += 8
    l
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

  def skipByte() {
    ensure(1)
    off += 1
  }

  def skipInt() {
    ensure(4)
    off += 4
  }

  def skipLong() {
    ensure(8)
    off += 8
  }

  def skipFloat() {
    ensure(4)
    off += 4
  }

  def skipDouble() {
    ensure(8)
    off += 8
  }

  def skipBytes(n0: Int) {
    var n = n0
    while (n > 0) {
      if (end == off)
        readBlock()
      val p = math.min(end - off, n)
      n -= p
      off += p
    }
  }

  def readDoubles(to: Array[Double], toOff0: Int, n0: Int) {
    assert(toOff0 >= 0)
    assert(n0 >= 0)
    assert(toOff0 <= to.length - n0)
    var toOff = toOff0
    var n = n0

    while (n > 0) {
      if (end == off)
        readBlock()
      val p = math.min(end - off, n << 3) >>> 3
      assert(p > 0)
      Memory.memcpy(to, toOff, buf, off, p)
      toOff += p
      n -= p
      off += (p << 3)
    }
  }
}

trait Decoder extends Closeable {
  def close()

  def readRegionValue(region: Region): Long

  def readByte(): Byte
}

class MethodBuilderSelfLike(val mb: MethodBuilder) extends MethodBuilderLike[MethodBuilderSelfLike] {
  type MB = MethodBuilder

  def newMethod(paramInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilderSelfLike =
    new MethodBuilderSelfLike(mb.fb.newMethod(paramInfo, returnInfo))
}

object EmitPackDecoder {
  self =>

  type Emitter = EstimableEmitter[MethodBuilderSelfLike]

  def emitTypeSize(t: PType): Int = {
    t match {
      case t: PArray => 120 + emitTypeSize(t.elementType)
      case t: PStruct => 100
      case _ => 20
    }
  }

  def emitBinary(
    t: PBinary,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    val length = mb.newLocal[Int]
    val boff = mb.newLocal[Long]

    Code(
      length := in.readInt(),
      boff := srvb.allocateBinary(length),
      in.readBytes(srvb.region, boff + const(4), length))
  }

  def emitBaseStruct(
    t: PBaseStruct,
    requestedType: PBaseStruct,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    val region = srvb.region

    val moff = mb.newField[Long]

    val initCode = Code(
      srvb.start(init = true),
      moff := region.allocate(const(1), const(t.nMissingBytes)),
      in.readBytes(region, moff, t.nMissingBytes))

    val fieldEmitters = new Array[Emitter](t.size)

    assert(t.isInstanceOf[PTuple] || t.isInstanceOf[PStruct])

    var i = 0
    var j = 0
    while (i < t.size) {
      val f = t.fields(i)
      fieldEmitters(i) =
        if (t.isInstanceOf[PTuple] ||
          (j < requestedType.size && requestedType.fields(j).name == f.name)) {
          val rf = requestedType.fields(j)
          assert(f.typ.required == rf.typ.required)
          j += 1

          new Emitter {
            def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
              val readElement = self.emit(f.typ, rf.typ, mbLike.mb, in, srvb)
              Code(
                if (f.typ.required)
                  readElement
                else {
                  region.loadBit(moff, const(t.missingIdx(f.index))).mux(
                    srvb.setMissing(),
                    readElement)
                },
                srvb.advance())
            }

            def estimatedSize: Int = emitTypeSize(f.typ)
          }
        } else {
          new Emitter {
            def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
              val skipField = skip(f.typ, mbLike.mb, in, region)
              if (f.typ.required)
                skipField
              else {
                region.loadBit(moff, const(t.missingIdx(f.index))).mux(
                  Code._empty,
                  skipField)
              }
            }

            def estimatedSize: Int = emitTypeSize(f.typ)
          }
        }
      i += 1
    }
    assert(j == requestedType.size)

    Code(initCode,
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)),
      Code._empty)
  }

  def emitArray(
    t: PArray,
    requestedType: PArray,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    val length = mb.newLocal[Int]
    val i = mb.newLocal[Int]
    val aoff = mb.newLocal[Long]

    Code(
      length := in.readInt(),
      srvb.start(length, init = false),
      aoff := srvb.offset,
      srvb.region.storeInt(aoff, length),
      if (t.elementType.required)
        Code._empty
      else
        in.readBytes(srvb.region, aoff + const(4), (length + 7) >>> 3),
      i := 0,
      Code.whileLoop(
        i < length,
        Code({
          val readElement = emit(t.elementType, requestedType.elementType, mb, in, srvb)
          if (t.elementType.required)
            readElement
          else
            t.isElementDefined(srvb.region, aoff, i).mux(
              readElement,
              srvb.setMissing())
        },
          srvb.advance(),
          i := i + const(1))))
  }

  def skipBaseStruct(t: PBaseStruct, mb: MethodBuilder, in: Code[InputBuffer], region: Code[Region]): Code[Unit] = {
    val moff = mb.newField[Long]

    val fieldEmitters = t.fields.map { f =>
      new Emitter {
        def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
          val skipField = skip(f.typ, mbLike.mb, in, region)
          if (f.typ.required)
            skipField
          else
            region.loadBit(moff, const(t.missingIdx(f.index))).mux(
              Code._empty,
              skipField)
        }

        def estimatedSize: Int = emitTypeSize(f.typ)
      }
    }

    Code(
      moff := region.allocate(const(1), const(t.nMissingBytes)),
      in.readBytes(region, moff, t.nMissingBytes),
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)))
  }

  def skipArray(t: PArray,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    region: Code[Region]): Code[Unit] = {
    val length = mb.newLocal[Int]
    val i = mb.newLocal[Int]

    if (t.elementType.required) {
      Code(
        length := in.readInt(),
        i := 0,
        Code.whileLoop(i < length,
          Code(
            skip(t.elementType, mb, in, region),
            i := i + const(1))))
    } else {
      val moff = mb.newLocal[Long]
      val nMissing = mb.newLocal[Int]
      Code(
        length := in.readInt(),
        nMissing := ((length + 7) >>> 3),
        moff := region.allocate(const(1), nMissing.toL),
        in.readBytes(region, moff, nMissing),
        i := 0,
        Code.whileLoop(i < length,
          region.loadBit(moff, i.toL).mux(
            Code._empty,
            skip(t.elementType, mb, in, region)),
          i := i + const(1)))
    }
  }

  def skipBinary(t: PType, mb: MethodBuilder, in: Code[InputBuffer]): Code[Unit] = {
    val length = mb.newLocal[Int]
    Code(
      length := in.readInt(),
      in.skipBytes(length))
  }

  def skip(t: PType, mb: MethodBuilder, in: Code[InputBuffer], region: Code[Region]): Code[Unit] = {
    t match {
      case t2: PBaseStruct =>
        skipBaseStruct(t2, mb, in, region)
      case t2: PArray =>
        skipArray(t2, mb, in, region)
      case _: PBoolean => in.skipBoolean()
      case _: PInt64 => in.skipLong()
      case _: PInt32 => in.skipInt()
      case _: PFloat32 => in.skipFloat()
      case _: PFloat64 => in.skipDouble()
      case t2: PBinary => skipBinary(t2, mb, in)
    }
  }

  def emit(
    t: PType,
    requestedType: PType,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    t match {
      case t2: PBaseStruct =>
        val requestedType2 = requestedType.asInstanceOf[PBaseStruct]
        srvb.addBaseStruct(requestedType2, { srvb2 =>
          emitBaseStruct(t2, requestedType2, mb, in, srvb2)
        })
      case t2: PArray =>
        val requestedType2 = requestedType.asInstanceOf[PArray]
        srvb.addArray(requestedType2, { srvb2 =>
          emitArray(t2, requestedType2, mb, in, srvb2)
        })
      case _: PBoolean => srvb.addBoolean(in.readBoolean())
      case _: PInt64 => srvb.addLong(in.readLong())
      case _: PInt32 => srvb.addInt(in.readInt())
      case _: PFloat32 => srvb.addFloat(in.readFloat())
      case _: PFloat64 => srvb.addDouble(in.readDouble())
      case t2: PBinary => emitBinary(t2, mb, in, srvb)
    }
  }

  def apply(t: PType, requestedType: PType): () => AsmFunction2[Region, InputBuffer, Long] = {
    val fb = new Function2Builder[Region, InputBuffer, Long]
    val mb = fb.apply_method
    val in = mb.getArg[InputBuffer](2).load()
    val srvb = new StagedRegionValueBuilder(mb, requestedType)

    var c = t.fundamentalType match {
      case t: PBaseStruct =>
        emitBaseStruct(t, requestedType.fundamentalType.asInstanceOf[PBaseStruct], mb, in, srvb)
      case t: PArray =>
        emitArray(t, requestedType.fundamentalType.asInstanceOf[PArray], mb, in, srvb)
    }

    mb.emit(Code(
      c,
      Code._return(srvb.end())))

    fb.result()
  }
}

case class NativeDecoderModule(
  modKey: String,
  modBinary: Array[Byte]) extends Serializable

final class NativePackDecoder(in: InputStream, module: NativeDecoderModule) extends Decoder {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(module.modKey, module.modBinary)
  private[this] val make_decoder = mod.findPtrFuncL1(st, "make_input_buffer")
  assert(st.ok, st.toString())
  private[this] val decode_row = mod.findLongFuncL2(st, "decode_row")
  assert(st.ok, st.toString())
  private[this] val decode_byte = mod.findLongFuncL1(st, "decode_byte")
  assert(st.ok, st.toString())
  private[this] val input = new ObjectArray(in)
  private[this] val decoder = new NativePtr(make_decoder, st, input.get())
  input.close()
  make_decoder.close()
  assert(st.ok, st.toString())

  def close(): Unit = {
    decoder.close()
    decode_row.close()
    decode_byte.close()
    // NativePtr's to objects with destructors using the module code must
    // *not* be close'd last, since the module will be dlclose'd before the
    // destructor is called.  One safe policy is to close everything in
    // reverse order, ending with the NativeModule
    mod.close()
    st.close()
    in.close()
  }

  def readByte(): Byte = decode_byte(st, decoder.get()).toByte

  def readRegionValue(region: Region): Long = decode_row(st, decoder.get(), region.get())
}

final class CompiledPackDecoder(in: InputBuffer, f: () => AsmFunction2[Region, InputBuffer, Long]) extends Decoder {
  def close() {
    in.close()
  }

  def readByte(): Byte = in.readByte()

  def readRegionValue(region: Region): Long = {
    f()(region, in)
  }
}

final class PackDecoder(rowType: PType, in: InputBuffer) extends Decoder {
  def close() {
    in.close()
  }

  def readByte(): Byte = in.readByte()

  def readBinary(region: Region, off: Long) {
    val length = in.readInt()
    val boff = region.allocate(4, 4 + length)
    region.storeAddress(off, boff)
    region.storeInt(boff, length)
    in.readBytes(region, boff + 4, length)
  }

  def readArray(t: PArray, region: Region): Long = {
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

    if (t.elementType == PInt32Required) { // fast path
      var i = 0
      while (i < length) {
        val off = elemsOff + i * elemSize
        region.storeInt(off, in.readInt())
        i += 1
      }
    } else {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          t.elementType match {
            case t2: PBaseStruct => readBaseStruct(t2, region, off)
            case t2: PArray =>
              val aoff = readArray(t2, region)
              region.storeAddress(off, aoff)
            case _: PBoolean => region.storeByte(off, in.readBoolean().toByte)
            case _: PInt64 => region.storeLong(off, in.readLong())
            case _: PInt32 => region.storeInt(off, in.readInt())
            case _: PFloat32 => region.storeFloat(off, in.readFloat())
            case _: PFloat64 => region.storeDouble(off, in.readDouble())
            case _: PBinary => readBinary(region, off)
          }
        }
        i += 1
      }
    }

    aoff
  }

  def readBaseStruct(t: PBaseStruct, region: Region, offset: Long) {
    val nMissingBytes = t.nMissingBytes
    in.readBytes(region, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(region, offset, i)) {
        val off = offset + t.byteOffsets(i)
        t.types(i) match {
          case t2: PBaseStruct => readBaseStruct(t2, region, off)
          case t2: PArray =>
            val aoff = readArray(t2, region)
            region.storeAddress(off, aoff)
          case _: PBoolean => region.storeByte(off, in.readBoolean().toByte)
          case _: PInt32 => region.storeInt(off, in.readInt())
          case _: PInt64 => region.storeLong(off, in.readLong())
          case _: PFloat32 => region.storeFloat(off, in.readFloat())
          case _: PFloat64 => region.storeDouble(off, in.readDouble())
          case _: PBinary => readBinary(region, off)
        }
      }
      i += 1
    }
  }

  def readRegionValue(region: Region): Long = {
    rowType.fundamentalType match {
      case t: PBaseStruct =>
        val start = region.allocate(t.alignment, t.byteSize)
        readBaseStruct(t, region, start)
        start

      case t: PArray =>
        readArray(t, region)
    }
  }
}

trait Encoder extends Closeable {
  def flush(): Unit

  def close(): Unit

  def writeRegionValue(region: Region, offset: Long): Unit

  def writeByte(b: Byte): Unit
}

object EmitPackEncoder { self =>

  type Emitter = EstimableEmitter[MethodBuilderSelfLike]

  def emitTypeSize(t: PType): Int = {
    t match {
      case t: PArray => 120 + emitTypeSize(t.elementType)
      case t: PStruct => 100
      case _ => 20
    }
  }

  def emitStruct(t: PStruct, requestedType: PStruct, mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val writeMissingBytes =
      if (requestedType.size == t.size)
        out.writeBytes(region, off, t.nMissingBytes)
      else {
        var c: Code[Unit] = Code._empty[Unit]
        var j = 0
        var n = 0
        while (j < requestedType.size) {
          var b = const(0)
          var k = 0
          while (k < 8 && j < requestedType.size) {
            val rf = requestedType.fields(j)
            if (!rf.typ.required) {
              val i = t.fieldIdx(rf.name)
              b = b | (t.isFieldMissing(region, off, i).toI << k)
              k += 1
            }
            j += 1
          }
          if (k > 0) {
            c = Code(c, out.writeByte(b.toB))
            n += 1
          }
        }
        assert(n == requestedType.nMissingBytes)
        c
      }

    val foff = mb.newField[Long]

    val fieldEmitters = requestedType.fields.zipWithIndex.map { case (rf, j) =>
      val i = t.fieldIdx(rf.name)
      val f = t.fields(i)
      new Emitter {
        def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
          val mb = mbLike.mb
          val region = mb.getArg[Region](1).load()
          val offset = mb.getArg[Long](2).load()
          val out = mb.getArg[OutputBuffer](3).load()

          t.isFieldDefined(region, foff, i).mux(
            self.emit(f.typ, rf.typ, mb, region, t.fieldOffset(foff, i), out),
            Code._empty[Unit])
        }

        def estimatedSize: Int = emitTypeSize(rf.typ)
      }
    }

    Code(
      writeMissingBytes,
      foff := off,
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)),
      Code._empty[Unit])
  }

  def emitTuple(t: PTuple, requestedType: PTuple, mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val foff = mb.newField[Long]

    val writeMissingBytes = out.writeBytes(region, off, t.nMissingBytes)

    val fieldEmitters = t.types.zipWithIndex.map { case (ft, i) =>
      new Emitter {
        def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
          val mb = mbLike.mb
          val region = mb.getArg[Region](1).load()
          val offset = mb.getArg[Long](2).load()
          val out = mb.getArg[OutputBuffer](3).load()

          t.isFieldDefined(region, foff, i).mux(
            self.emit(ft, requestedType.types(i), mb, region, t.fieldOffset(foff, i), out),
            Code._empty[Unit])
        }

        def estimatedSize: Int = emitTypeSize(ft)
      }
    }

    Code(
      writeMissingBytes,
      foff := off,
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)),
      Code._empty[Unit])
  }

  def emitArray(t: PArray, requestedType: PArray, mb: MethodBuilder, region: Code[Region], aoff: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val length = region.loadInt(aoff)

    val writeLen = out.writeInt(length)
    val writeMissingBytes =
      if (!t.elementType.required) {
        val nMissingBytes = (length + 7) >>> 3
        out.writeBytes(region, aoff + const(4), nMissingBytes)
      } else
        Code._empty[Unit]

    val i = mb.newLocal[Int]("i")

    val writeElems = Code(
      i := 0,
      Code.whileLoop(
        i < length,
        Code(t.isElementDefined(region, aoff, i).mux(
          emit(t.elementType, requestedType.elementType, mb, region, t.elementOffset(aoff, length, i), out),
          Code._empty[Unit]),
          i := i + const(1))))

    Code(writeLen, writeMissingBytes, writeElems)
  }

  def writeBinary(mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val boff = region.loadAddress(off)
    val length = region.loadInt(boff)
    Code(out.writeInt(length),
      out.writeBytes(region, boff + const(4), length))
  }

  def emit(t: PType, requestedType: PType, mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    t.fundamentalType match {
      case t: PStruct => emitStruct(t, requestedType.fundamentalType.asInstanceOf[PStruct], mb, region, off, out)
      case t: PTuple => emitTuple(t, requestedType.fundamentalType.asInstanceOf[PTuple], mb, region, off, out)
      case t: PArray => emitArray(t, requestedType.fundamentalType.asInstanceOf[PArray], mb, region, region.loadAddress(off), out)
      case _: PBoolean => out.writeBoolean(region.loadBoolean(off))
      case _: PInt32 => out.writeInt(region.loadInt(off))
      case _: PInt64 => out.writeLong(region.loadLong(off))
      case _: PFloat32 => out.writeFloat(region.loadFloat(off))
      case _: PFloat64 => out.writeDouble(region.loadDouble(off))
      case _: PBinary => writeBinary(mb, region, off, out)
    }
  }

  def apply(t: PType, requestedType: PType): () => AsmFunction3[Region, Long, OutputBuffer, Unit] = {
    val fb = new Function3Builder[Region, Long, OutputBuffer, Unit]
    val mb = fb.apply_method
    val region = mb.getArg[Region](1).load()
    val offset = mb.getArg[Long](2).load()
    val out = mb.getArg[OutputBuffer](3).load()

    mb.emit(emit(t, requestedType, mb, region, offset, out))
    fb.result()
  }
}

final class CompiledPackEncoder(out: OutputBuffer, f: () => AsmFunction3[Region, Long, OutputBuffer, Unit]) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeRegionValue(region: Region, offset: Long) {
    f()(region, offset, out)
  }

  def writeByte(b: Byte) {
    out.writeByte(b)
  }
}

final class PackEncoder(rowType: PType, out: OutputBuffer) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeByte(b: Byte): Unit = out.writeByte(b)

  def writeBinary(region: Region, offset: Long) {
    val boff = region.loadAddress(offset)
    val length = region.loadInt(boff)
    out.writeInt(length)
    out.writeBytes(region, boff + 4, length)
  }

  def writeArray(t: PArray, region: Region, aoff: Long) {
    val length = region.loadInt(aoff)

    out.writeInt(length)
    if (!t.elementType.required) {
      val nMissingBytes = (length + 7) >>> 3
      out.writeBytes(region, aoff + 4, nMissingBytes)
    }

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = t.elementByteSize
    if (t.elementType.isInstanceOf[PInt32]) { // fast case
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
            case t2: PBaseStruct => writeBaseStruct(t2, region, off)
            case t2: PArray => writeArray(t2, region, region.loadAddress(off))
            case _: PBoolean => out.writeBoolean(region.loadByte(off) != 0)
            case _: PInt64 => out.writeLong(region.loadLong(off))
            case _: PFloat32 => out.writeFloat(region.loadFloat(off))
            case _: PFloat64 => out.writeDouble(region.loadDouble(off))
            case _: PBinary => writeBinary(region, off)
          }
        }

        i += 1
      }
    }
  }

  def writeBaseStruct(t: PBaseStruct, region: Region, offset: Long) {
    val nMissingBytes = t.nMissingBytes
    out.writeBytes(region, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(region, offset, i)) {
        val off = offset + t.byteOffsets(i)
        t.types(i) match {
          case t2: PBaseStruct => writeBaseStruct(t2, region, off)
          case t2: PArray => writeArray(t2, region, region.loadAddress(off))
          case _: PBoolean => out.writeBoolean(region.loadByte(off) != 0)
          case _: PInt32 => out.writeInt(region.loadInt(off))
          case _: PInt64 => out.writeLong(region.loadLong(off))
          case _: PFloat32 => out.writeFloat(region.loadFloat(off))
          case _: PFloat64 => out.writeDouble(region.loadDouble(off))
          case _: PBinary => writeBinary(region, off)
        }
      }

      i += 1
    }
  }

  def writeRegionValue(region: Region, offset: Long) {
    (rowType.fundamentalType: @unchecked) match {
      case t: PBaseStruct =>
        writeBaseStruct(t, region, offset)
      case t: PArray =>
        writeArray(t, region, offset)
    }
  }
}

case class NativeEncoderModule(
  modKey: String,
  modBinary: Array[Byte]) extends Serializable

final class NativePackEncoder(out: OutputStream, module: NativeEncoderModule) extends Encoder {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(module.modKey, module.modBinary)
  private[this] val makeOutputBufferF = mod.findPtrFuncL1(st, "makeOutputBuffer")
  assert(st.ok, st.toString())
  private[this] val encodeByteF = mod.findLongFuncL2(st, "encode_byte")
  assert(st.ok, st.toString())
  private[this] val encodeRVF = mod.findLongFuncL2(st, "encode_row")
  assert(st.ok, st.toString())
  private[this] val flushF = mod.findLongFuncL1(st, "encoder_flush")
  assert(st.ok, st.toString())
  private[this] val closeF = mod.findLongFuncL1(st, "encoder_close")
  assert(st.ok, st.toString())

  private[this] val objArray = new ObjectArray(out)
  assert(st.ok, st.toString())
  val buf = new NativePtr(makeOutputBufferF, st, objArray.get())
  objArray.close()
  makeOutputBufferF.close()

  def flush(): Unit = {
    flushF(st, buf.get())
    assert(st.ok, st.toString())
    out.flush()
  }

  def close(): Unit = {
    closeF(st, buf.get())
    assert(st.ok, st.toString())
    buf.close()
    encodeByteF.close()
    encodeRVF.close()
    flushF.close()
    closeF.close()
    mod.close()
    st.close()
    out.close()
  }

  def writeRegionValue(region: Region, offset: Long): Unit = {
    encodeRVF(st, buf.get(), offset)
    assert(st.ok, st.toString())
  }

  def writeByte(b: Byte): Unit = {
    encodeByteF(st, buf.get(), b)
    assert(st.ok, st.toString())
  }
}

object RichContextRDDRegionValue {
  def writeRowsPartition(makeEnc: (OutputStream) => Encoder)(ctx: RVDContext, it: Iterator[RegionValue], os: OutputStream): Long = {
    val context = TaskContext.get
    val outputMetrics =
      if (context != null)
        context.taskMetrics().outputMetrics
      else
        null
    val trackedOS = new ByteTrackingOutputStream(os)
    val en = makeEnc(trackedOS)
    var rowCount = 0L

    it.foreach { rv =>
      en.writeByte(1)
      en.writeRegionValue(rv.region, rv.offset)
      ctx.region.clear()
      rowCount += 1

      if (outputMetrics != null) {
        ExposedMetrics.setBytes(outputMetrics, trackedOS.bytesWritten)
        ExposedMetrics.setRecords(outputMetrics, rowCount)
      }
    }

    en.writeByte(0) // end
    en.flush()
    if (outputMetrics != null) {
      ExposedMetrics.setBytes(outputMetrics, trackedOS.bytesWritten)
    }
    os.close()

    rowCount
  }

  def writeSplitRegion(
    fs: FS,
    path: String,
    t: RVDType,
    it: Iterator[RegionValue],
    idx: Int,
    ctx: RVDContext,
    partDigits: Int,
    stageLocally: Boolean,
    makeRowsEnc: (OutputStream) => Encoder,
    makeEntriesEnc: (OutputStream) => Encoder
  ): (String, Long) = {
    val fullRowType = t.rowType

    val context = TaskContext.get
    val f = partFile(partDigits, idx, context)
    val outputMetrics = context.taskMetrics().outputMetrics
    val finalRowsPartPath = path + "/rows/rows/parts/" + f
    val finalEntriesPartPath = path + "/entries/rows/parts/" + f
    val (rowsPartPath, entriesPartPath) =
      if (stageLocally) {
        val rowsPartPath = fs.getTemporaryFile("file:///tmp")
        val entriesPartPath = fs.getTemporaryFile("file:///tmp")
        context.addTaskCompletionListener { (context: TaskContext) =>
          fs.delete(rowsPartPath, recursive = false)
          fs.delete(entriesPartPath, recursive = false)
        }
        (rowsPartPath, entriesPartPath)
      } else
        (finalRowsPartPath, finalEntriesPartPath)

    val rowCount = fs.writeFile(rowsPartPath) { rowsOS =>
      val trackedRowsOS = new ByteTrackingOutputStream(rowsOS)
      using(makeRowsEnc(trackedRowsOS)) { rowsEN =>

        fs.writeFile(entriesPartPath) { entriesOS =>
          val trackedEntriesOS = new ByteTrackingOutputStream(entriesOS)
          using(makeEntriesEnc(trackedEntriesOS)) { entriesEN =>

            var rowCount = 0L

            it.foreach { rv =>
              rowsEN.writeByte(1)
              rowsEN.writeRegionValue(rv.region, rv.offset)

              entriesEN.writeByte(1)
              entriesEN.writeRegionValue(rv.region, rv.offset)

              ctx.region.clear()

              rowCount += 1

              ExposedMetrics.setBytes(outputMetrics, trackedRowsOS.bytesWritten + trackedEntriesOS.bytesWritten)
              ExposedMetrics.setRecords(outputMetrics, 2 * rowCount)
            }

            rowsEN.writeByte(0) // end
            entriesEN.writeByte(0)

            rowsEN.flush()
            entriesEN.flush()
            ExposedMetrics.setBytes(outputMetrics, trackedRowsOS.bytesWritten + trackedEntriesOS.bytesWritten)

            rowCount
          }
        }
      }
    }

    if (stageLocally) {
      fs.copy(rowsPartPath, finalRowsPartPath)
      fs.copy(entriesPartPath, finalEntriesPartPath)
    }

    f -> rowCount
  }

  def writeSplitSpecs(
    fs: FS,
    path: String,
    codecSpec: CodecSpec,
    key: IndexedSeq[String],
    rowsRVType: PStruct,
    entriesRVType: PStruct,
    partFiles: Array[String],
    partitioner: RVDPartitioner
  ) {
    val rowsSpec = OrderedRVDSpec(rowsRVType, key, codecSpec, partFiles, partitioner)
    rowsSpec.write(fs, path + "/rows/rows")

    val entriesSpec = OrderedRVDSpec(entriesRVType, FastIndexedSeq(), codecSpec, partFiles, RVDPartitioner.unkeyed(partitioner.numPartitions))
    entriesSpec.write(fs, path + "/entries/rows")
  }
}

class RichContextRDDRegionValue(val crdd: ContextRDD[RVDContext, RegionValue]) extends AnyVal {
  def boundary: ContextRDD[RVDContext, RegionValue] =
    crdd.cmapPartitionsAndContext { (consumerCtx, part) =>
      val producerCtx = consumerCtx.freshContext
      val it = part.flatMap(_ (producerCtx))
      new Iterator[RegionValue]() {
        private[this] var cleared: Boolean = false

        def hasNext: Boolean = {
          if (!cleared) {
            cleared = true
            producerCtx.region.clear()
          }
          it.hasNext
        }

        def next: RegionValue = {
          if (!cleared) {
            producerCtx.region.clear()
          }
          cleared = false
          it.next
        }
      }
    }

  def writeRows(path: String, t: PStruct, stageLocally: Boolean, codecSpec: CodecSpec): (Array[String], Array[Long]) = {
    crdd.writePartitions(path, stageLocally, RichContextRDDRegionValue.writeRowsPartition(codecSpec.buildEncoder(t)))
  }

  def writeRowsSplit(
    path: String,
    t: RVDType,
    codecSpec: CodecSpec,
    partitioner: RVDPartitioner,
    stageLocally: Boolean
  ): Array[Long] = {
    val fs = HailContext.sFS

    fs.mkDir(path + "/rows/rows/parts")
    fs.mkDir(path + "/entries/rows/parts")

    val bcFS = HailContext.bcFS

    val nPartitions = crdd.getNumPartitions
    val d = digitsNeeded(nPartitions)

    val fullRowType = t.rowType
    val rowsRVType = MatrixType.getRowType(fullRowType)
    val entriesRVType = MatrixType.getSplitEntriesType(fullRowType)

    val makeRowsEnc = codecSpec.buildEncoder(fullRowType, rowsRVType)

    val makeEntriesEnc = codecSpec.buildEncoder(fullRowType, entriesRVType)

    val partFilePartitionCounts = crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
      val fs = bcFS.value

      val partFileAndCount = RichContextRDDRegionValue.writeSplitRegion(
        fs,
        path,
        t,
        it,
        i,
        ctx,
        d,
        stageLocally,
        makeRowsEnc,
        makeEntriesEnc)

      Iterator.single(partFileAndCount)
    }.collect()

    val (partFiles, partitionCounts) = partFilePartitionCounts.unzip

    RichContextRDDRegionValue.writeSplitSpecs(fs, path, codecSpec, t.key, rowsRVType, entriesRVType, partFiles, partitioner)

    partitionCounts
  }

  def toRows(rowType: PStruct): RDD[Row] = {
    crdd.run.map(rv => SafeRow(rowType, rv.region, rv.offset))
  }
}
