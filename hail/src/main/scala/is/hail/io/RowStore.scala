package is.hail.io

import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.io.compress.LZ4Utils
import is.hail.nativecode._
import is.hail.rvd.{AbstractRVDSpec, RVDContext, RVDPartitioner}
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.json4s.{Extraction, JValue}
import org.json4s.jackson.JsonMethods
import java.io._

import scala.collection.mutable.ArrayBuffer
import is.hail.asm4s._
import is.hail.cxx
import is.hail.expr.ir.{EmitUtils, EstimableEmitter, MethodBuilderLike}
import is.hail.expr.types.MatrixType
import is.hail.expr.types.physical._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.apache.spark.sql.Row
import org.apache.spark.{ExposedMetrics, TaskContext}

trait BufferSpec extends Serializable {
  def buildInputBuffer(in: InputStream): InputBuffer

  def buildOutputBuffer(out: OutputStream): OutputBuffer

  def nativeOutputBufferType: String

  def nativeInputBufferType: String
}

final class LEB128BufferSpec(child: BufferSpec) extends BufferSpec {
  def buildInputBuffer(in: InputStream): InputBuffer = new LEB128InputBuffer(child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new LEB128OutputBuffer(child.buildOutputBuffer(out))

  def nativeOutputBufferType: String = s"LEB128OutputBuffer<${ child.nativeOutputBufferType }>"

  def nativeInputBufferType: String = s"LEB128InputBuffer<${ child.nativeInputBufferType }>"
}

final class BlockingBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BufferSpec {
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

final class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BlockBufferSpec {
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

  val blockSpecs: Array[BufferSpec] = Array(
    new BlockingBufferSpec(64 * 1024,
      new StreamBlockBufferSpec),
    new BlockingBufferSpec(32 * 1024,
      new LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)))

  val bufferSpecs: Array[BufferSpec] = blockSpecs.flatMap { blockSpec =>
    Array(blockSpec,
      new LEB128BufferSpec(blockSpec))
  }

  val codecSpecs: Array[CodecSpec] = bufferSpecs.flatMap { bufferSpec =>
    Array(new PackCodecSpec(bufferSpec))
  }

  val supportedCodecSpecs: Array[CodecSpec] = bufferSpecs.flatMap { bufferSpec =>
    Array(new PackCodecSpec(bufferSpec))
  }
}

trait CodecSpec extends Serializable {
  def buildEncoder(t: PType): (OutputStream) => Encoder

  def buildDecoder(t: PType, requestedType: PType): (InputStream) => Decoder

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

  def buildEncoder(t: PType): (OutputStream) => Encoder = {
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null) {
      val e: NativeEncoderModule = NativeEncoder(t, child)
      (out: OutputStream) => new NativePackEncoder(out, e)
    } else {
      { out: OutputStream => new PackEncoder(t, child.buildOutputBuffer(out)) }
    }
  }

  def buildDecoder(t: PType, requestedType: PType): (InputStream) => Decoder = {
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null) {
      val d: NativeDecoderModule = NativeDecoder(t, requestedType, child)
      (in: InputStream) => new NativePackDecoder(in, d)
    } else {
      val f = EmitPackDecoder(t, requestedType)
      (in: InputStream) => new CompiledPackDecoder(child.buildInputBuffer(in), f)
    }
  }
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
  def decoderId: Int

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

final class LEB128InputBuffer(in: InputBuffer) extends InputBuffer {
  def close() {
    in.close()
  }

  def decoderId = 1

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

  def decoderId = 0

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

object NativeDecoder {
  def skipBinary(input_buf_ptr: cxx.Expression): cxx.Code = {
    val len = cxx.Variable("skip_len", "int", s"$input_buf_ptr->read_int()")
    s"""
       |${ len.define }
       |$input_buf_ptr->skip_bytes($len);""".stripMargin
  }

  def skipArray(t: PArray, input_buf_ptr: cxx.Expression): cxx.Code = {
    val len = cxx.Variable("len", "int", s"$input_buf_ptr->read_int()")
    val i = cxx.Variable("i", "int", "0")
    if (t.elementType.required) {
      s"""
         |${ len.define }
         |for (${ i.define } $i < $len; $i++) {
         |  ${ skip(t.elementType, input_buf_ptr) }
         |}""".stripMargin
    } else {
      val missingBytes = cxx.ArrayVariable(s"missing", "char", s"n_missing_bytes($len)")
      s"""
         |${ len.define }
         |${ missingBytes.define }
         |$input_buf_ptr->read_bytes($missingBytes, n_missing_bytes($len));
         |for (${ i.define } $i < $len; $i++) {
         |  if (!load_bit($missingBytes, $i)) {
         |    ${ skip(t.elementType, input_buf_ptr) }
         |  }
         |}""".stripMargin
    }
  }

  def skipBaseStruct(t: PBaseStruct, input_buf_ptr: cxx.Expression): cxx.Code = {
    val missingBytes = cxx.ArrayVariable("missing", "char", s"${ t.nMissingBytes }")
    val skipFields = Array.tabulate[cxx.Code](t.size) { idx =>
      val fieldType = t.types(idx)
      if (fieldType.required)
        skip(fieldType, input_buf_ptr)
      else
        s"""
           |if (!load_bit($missingBytes, ${ t.missingIdx(idx) })) {
           |  ${ skip(fieldType, input_buf_ptr) }
           |}""".stripMargin
    }

    if (t.nMissingBytes > 0)
      s"""
         |${ missingBytes.define }
         |$input_buf_ptr->read_bytes($missingBytes, ${ t.nMissingBytes });
         |${ skipFields.mkString("\n") }""".stripMargin
    else
      skipFields.mkString("\n")
  }

  def skip(t: PType, input_buf_ptr: cxx.Expression): cxx.Code = t match {
    case t2: PArray => skipArray(t2, input_buf_ptr)
    case t2: PBaseStruct => skipBaseStruct(t2, input_buf_ptr)
    case _: PBinary => skipBinary(input_buf_ptr)
    case _: PBoolean => s"$input_buf_ptr->skip_boolean();"
    case _: PInt32 => s"$input_buf_ptr->skip_int();"
    case _: PInt64 => s"$input_buf_ptr->skip_long();"
    case _: PFloat32 => s"$input_buf_ptr->skip_float();"
    case _: PFloat64 => s"$input_buf_ptr->skip_double();"
  }

  def decodeBinary(input_buf_ptr: cxx.Expression, region: cxx.Expression, off: cxx.Expression, fb: cxx.FunctionBuilder): cxx.Code = {
    val len = cxx.Variable("len", "int", s"$input_buf_ptr->read_int()")
    val boff = cxx.Variable("boff", "char *", s"$region->allocate(${ PBinary.contentAlignment }, $len + 4)")
    s"""
       |${ len.define }
       |${ boff.define }
       |store_address($off, $boff);
       |store_int($boff, $len);
       |$input_buf_ptr->read_bytes($boff + 4, $len);""".stripMargin
  }

  def decodeArray(t: PArray, rt: PArray, input_buf_ptr: cxx.Expression, region: cxx.Expression, off: cxx.Expression, fb: cxx.FunctionBuilder): cxx.Code = {
    val len = cxx.Variable("len", "int", s"$input_buf_ptr->read_int()")
    val sab = new cxx.StagedContainerBuilder(fb, region.toString, rt)
    var decodeElt = decode(t.elementType, rt.elementType, input_buf_ptr, region, cxx.Expression(sab.eltOffset), fb)
    if (rt.elementType.required)
      decodeElt =
        s"""
           |if (!${ rt.cxxIsElementMissing(sab.end(), sab.idx) }) {
           |  $decodeElt
           |}
         """.stripMargin

    s"""
       |${ len.define }
       |${ sab.start(len, clearMissing = false) }
       |store_address($off, ${ sab.end() });
       |${ if (rt.elementType.required) "" else s"$input_buf_ptr->read_bytes(${ sab.end() } + 4, ${ rt.cxxNMissingBytes(s"$len") });" }
       |while (${ sab.idx } < $len) {
       |  $decodeElt
       |  ${ sab.advance() }
       |}
     """.stripMargin
  }

  def decodeBaseStruct(t: PBaseStruct, rt: PBaseStruct, input_buf_ptr: cxx.Expression, region: cxx.Expression, off: cxx.Expression, fb: cxx.FunctionBuilder): cxx.Code = {
    val ssb = new cxx.StagedBaseStructBuilder(fb, rt, off)

    def wrapMissing(missingBytes: cxx.Code, tidx: Int)(processField: cxx.Code): cxx.Code = {
      if (!t.fieldRequired(tidx)) {
        s"""
           |if (!${ t.cxxIsFieldMissing(missingBytes, tidx) }) {
           |  $processField
           |}""".stripMargin
      } else processField
    }

    if (t.size == rt.size) {
      assert((t.isInstanceOf[PTuple] && rt.isInstanceOf[PTuple]) || (t.asInstanceOf[PStruct].fieldNames sameElements t.asInstanceOf[PStruct].fieldNames))
      val decodeFields = Array.tabulate[cxx.Code](rt.size) { idx =>
        wrapMissing(s"$off", idx)(ssb.addField(idx, foff => decode(t.types(idx), rt.types(idx), input_buf_ptr, region, cxx.Expression(foff), fb)))
      }.mkString("\n")
      if (t.nMissingBytes > 0) {
        s"""
           |$input_buf_ptr->read_bytes($off, ${ t.nMissingBytes });
           |$decodeFields""".stripMargin
      } else
        decodeFields
    } else {
      val names = t.asInstanceOf[PStruct].fieldNames
      val rnames = rt.asInstanceOf[PStruct].fieldNames
      val t_missing_bytes = cxx.ArrayVariable(s"missing", "char", s"${ t.nMissingBytes }")

      var rtidx = 0
      val decodeFields = Array.tabulate[cxx.Code](t.size) { tidx =>
        val f = t.types(tidx)
        val skipField = rtidx >= rt.size || names(tidx) != rnames(rtidx)
        val processField = if (skipField)
          wrapMissing(s"$t_missing_bytes", tidx)(skip(f, input_buf_ptr))
        else
          s"""
             |${ wrapMissing(s"$t_missing_bytes", tidx)(ssb.addField(rtidx, foff => decode(f, rt.types(rtidx), input_buf_ptr, region, cxx.Expression(foff), fb))) }
             |${ if (f.required) "" else s" else { ${ ssb.setMissing(rtidx) } }" }
           """.stripMargin
        if (!skipField)
          rtidx += 1
        wrapMissing(s"$t_missing_bytes", tidx)(processField)
      }.mkString("\n")
      if (t.nMissingBytes > 0) {
        s"""
           |${ t_missing_bytes.define }
           |${ ssb.clearAllMissing() };
           |$input_buf_ptr->read_bytes($t_missing_bytes, ${ t.nMissingBytes });
           |$decodeFields""".stripMargin
      } else
        decodeFields
    }
  }

  def decode(t: PType, rt: PType, input_buf_ptr: cxx.Expression, region: cxx.Expression, off: cxx.Expression, fb: cxx.FunctionBuilder): cxx.Code = t match {
    case _: PBoolean => s"store_byte($off, $input_buf_ptr->read_byte());"
    case _: PInt32 => s"store_int($off, $input_buf_ptr->read_int());"
    case _: PInt64 => s"store_long($off, $input_buf_ptr->read_long());"
    case _: PFloat32 => s"store_float($off, $input_buf_ptr->read_float());"
    case _: PFloat64 => s"store_double($off, $input_buf_ptr->read_double());"
    case _: PBinary => decodeBinary(input_buf_ptr, region, off, fb)
    case t2: PArray => decodeArray(t2, rt.asInstanceOf[PArray], input_buf_ptr, region, off, fb)
    case t2: PBaseStruct => decodeBaseStruct(t2, rt.asInstanceOf[PBaseStruct], input_buf_ptr, region, off, fb)
  }

  def apply(t: PType, rt: PType, bufSpec: BufferSpec): NativeDecoderModule = {
    assert(t.isInstanceOf[PBaseStruct] || t.isInstanceOf[PArray])
    val tub = new cxx.TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Decoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("hail/Region.h")
    tub.include("hail/Utils.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val inBufFB = cxx.FunctionBuilder("make_input_buffer", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")
    inBufFB += "UpcallEnv up;"
    inBufFB += s"auto jinput_stream = reinterpret_cast<ObjectArray*>(${ inBufFB.getArg(1) })->at(0);"
    val bufType = bufSpec.nativeInputBufferType
    inBufFB += s"return std::make_shared<Decoder<$bufType>>(std::make_shared<InputStream>(up, jinput_stream));"
    tub += inBufFB.result()

    val rowFB = cxx.FunctionBuilder("decode_row", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "region"), "long")
    val buf = cxx.Variable("buf", s"$bufType *", s"&reinterpret_cast<Decoder<$bufType> *>(${ rowFB.getArg(1) })->buf_")
    val region = cxx.Variable("region", "Region *", s"reinterpret_cast<Region *>(${ rowFB.getArg(2) })")
    val initialSize = rt match {
      case _: PArray | _: PBinary => 8
      case _ => rt.byteSize
    }
    val row = cxx.Variable("row", "char *", s"$region->allocate(${ rt.alignment }, $initialSize)")
    rowFB += buf.define
    rowFB += region.define
    rowFB += row.define
    rowFB += decode(t.fundamentalType, rt.fundamentalType, buf.ref, region.ref, row.ref, rowFB)
    rowFB += (rt match {
      case _: PArray | _: PBinary => s"return read_long($row);"
      case _ => s"return reinterpret_cast<long>($row);"
    })
    tub += rowFB.result()

    val byteFB = cxx.FunctionBuilder("decode_byte", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    byteFB += s"return reinterpret_cast<Decoder<$bufType> *>(${ byteFB.getArg(1) })->buf_.read_byte();"
    tub += byteFB.result()

    val mod = tub.result().build("-O2 -llz4")

    NativeDecoderModule(mod.getKey, mod.getBinary)
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

object NativeEncoder {

  def encodeBinary(output_buf_ptr: cxx.Expression, off: cxx.Expression): cxx.Code = {
    val len = cxx.Variable("len", "int", s"load_length($off)")
    s"""
       |${ len.define }
       |$output_buf_ptr->write_int($len);
       |$output_buf_ptr->write_bytes($off + 4, $len);""".stripMargin
  }

  def encodeArray(t: PArray, output_buf_ptr: cxx.Expression, off: cxx.Expression): cxx.Code = {
    val len = cxx.Variable("len", "int", s"load_length($off)")
    val i = cxx.Variable("i", "int", s"0")
    val copyLengthAndMissing = if (t.elementType.required)
      s"$output_buf_ptr->write_int($len);"
    else
      s"""
         |$output_buf_ptr->write_int($len);
         |$output_buf_ptr->write_bytes($off + 4, n_missing_bytes($len));""".stripMargin
    val eltOff = cxx.Variable("eoff",
      "char *",
      s"round_up_alignment(${ if (!t.elementType.required) s"$off + 4 + n_missing_bytes($len)" else s"$off + 4" }, ${ t.elementType.alignment })")
    val elt = t.elementType match {
      case (_: PBinary | _: PArray) => s"load_address($eltOff)"
      case _ => eltOff.toString
    }

    val writeElt = if (t.elementType.required)
      encode(t.elementType, output_buf_ptr, cxx.Expression(elt))
    else
      s"""
         |if (!load_bit($off + 4, $i)) {
         |  ${ encode(t.elementType, output_buf_ptr, cxx.Expression(elt)) };
         |}""".stripMargin

    s"""
       |${ len.define }
       |$copyLengthAndMissing
       |${ eltOff.define }
       |for (${ i.define } $i < $len; $i++) {
       |  $writeElt
       |  $eltOff += ${ t.elementByteSize };
       |}
      """.stripMargin
  }

  def encodeBaseStruct(t: PBaseStruct, output_buf_ptr: cxx.Expression, off: cxx.Expression): cxx.Code = {
    val nMissingBytes = t.nMissingBytes
    val storeFields: Array[cxx.Code] = Array.tabulate[cxx.Code](t.size) { idx =>
      val store = t.types(idx) match {
        case t2@(_: PArray | _: PBinary) =>
          encode(t2, output_buf_ptr, cxx.Expression(s"load_address($off + ${ t.byteOffsets(idx) })"))
        case t2 =>
          encode(t2, output_buf_ptr, cxx.Expression(s"$off + ${ t.byteOffsets(idx) }"))

      }
      if (t.fieldRequired(idx)) {
        store
      } else {
        s"""
           |if (!load_bit($off, ${ t.missingIdx(idx) })) {
           |  $store
           |}""".stripMargin
      }
    }
    s"""
       |$output_buf_ptr->write_bytes($off, $nMissingBytes);
       |${ storeFields.mkString("\n") }
      """.stripMargin
  }

  def encode(t: PType, output_buf_ptr: cxx.Expression, off: cxx.Expression): cxx.Code = t match {
    case _: PBoolean => s"$output_buf_ptr->write_byte(*($off) ? 1 : 0);"
    case _: PInt32 => s"$output_buf_ptr->write_int(load_int($off));"
    case _: PInt64 => s"$output_buf_ptr->write_long(load_long($off));"
    case _: PFloat32 => s"$output_buf_ptr->write_float(load_float($off));"
    case _: PFloat64 => s"$output_buf_ptr->write_double(load_double($off));"
    case _: PBinary => encodeBinary(output_buf_ptr, off)
    case t2: PArray => encodeArray(t2, output_buf_ptr, off)
    case t2: PBaseStruct => encodeBaseStruct(t2, output_buf_ptr, off)
  }

  def apply(t: PType, bufSpec: BufferSpec): NativeEncoderModule = {
    assert(t.isInstanceOf[PBaseStruct] || t.isInstanceOf[PArray])
    val tub = new cxx.TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Encoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("hail/Utils.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val outBufFB = cxx.FunctionBuilder("makeOutputBuffer", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")
    outBufFB += "UpcallEnv up;"
    outBufFB += s"auto joutput_stream = reinterpret_cast<ObjectArray*>(${ outBufFB.getArg(1) })->at(0);"
    val bufType = bufSpec.nativeOutputBufferType
    outBufFB += s"return std::make_shared<Encoder<$bufType>>(std::make_shared<OutputStream>(up, joutput_stream));"
    tub += outBufFB.result()

    val rowFB = cxx.FunctionBuilder("encode_row", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "row"), "long")
    val buf = cxx.Variable("buf", s"$bufType *", s"&reinterpret_cast<Encoder<$bufType> *>(${ rowFB.getArg(1) })->buf_")
    val row = cxx.Variable("row", "char *", s"reinterpret_cast<char *>(${ rowFB.getArg(2) })")
    rowFB += buf.define
    rowFB += row.define
    rowFB += encode(t.fundamentalType, buf.ref, row.ref)
    rowFB += "return 0;"
    tub += rowFB.result()

    val byteFB = cxx.FunctionBuilder("encode_byte", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "b"), "long")
    byteFB += s"reinterpret_cast<Encoder<$bufType> *>(${ byteFB.getArg(1) })->buf_.write_byte(${ byteFB.getArg(2) } & 0xff);"
    byteFB += "return 0;"
    tub += byteFB.result()

    val flushFB = cxx.FunctionBuilder("encoder_flush", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    flushFB += s"reinterpret_cast<Encoder<$bufType> *>(${ flushFB.getArg(1) })->buf_.flush();"
    flushFB += "return 0;"
    tub += flushFB.result()

    val closeFB = cxx.FunctionBuilder("encoder_close", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    closeFB +=
      s"""
         |reinterpret_cast<Encoder<$bufType> *>(${ closeFB.getArg(1) })->buf_.close();
         |return 0;""".stripMargin
    tub += closeFB.result()

    val mod = tub.result().build("-O1 -llz4")

    NativeEncoderModule(mod.getKey, mod.getBinary)
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
    t: MatrixType,
    codecSpec: CodecSpec,
    partitioner: RVDPartitioner,
    stageLocally: Boolean
  ): Array[Long] = {
    val sc = crdd.sparkContext
    val hConf = sc.hadoopConfiguration

    hConf.mkDir(path + "/rows/rows/parts")
    hConf.mkDir(path + "/entries/rows/parts")

    val sHConfBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))

    val nPartitions = crdd.getNumPartitions
    val d = digitsNeeded(nPartitions)

    val fullRowType = t.rvRowType
    val fullRowPType = fullRowType.physicalType
    val rowsRVType = t.rowType
    val localEntriesIndex = t.entriesIdx
    val entriesRVType = t.entriesRVType

    val makeRowsEnc = codecSpec.buildEncoder(rowsRVType.physicalType)

    val makeEntriesEnc = codecSpec.buildEncoder(t.entriesRVType.physicalType)

    val partFilePartitionCounts = crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
      val hConf = sHConfBc.value.value
      val context = TaskContext.get
      val f = partFile(d, i, context)
      val outputMetrics = context.taskMetrics().outputMetrics

      val finalRowsPartPath = path + "/rows/rows/parts/" + f
      val finalEntriesPartPath = path + "/entries/rows/parts/" + f

      val (rowsPartPath, entriesPartPath) =
        if (stageLocally) {
          val context = TaskContext.get
          val rowsPartPath = hConf.getTemporaryFile("file:///tmp")
          val entriesPartPath = hConf.getTemporaryFile("file:///tmp")
          context.addTaskCompletionListener { context =>
            hConf.delete(rowsPartPath, recursive = false)
            hConf.delete(entriesPartPath, recursive = false)
          }
          (rowsPartPath, entriesPartPath)
        } else
          (finalRowsPartPath, finalEntriesPartPath)

      val rowCount = hConf.writeFile(rowsPartPath) { rowsOS =>
        val trackedRowsOS = new ByteTrackingOutputStream(rowsOS)
        using(makeRowsEnc(trackedRowsOS)) { rowsEN =>

          hConf.writeFile(entriesPartPath) { entriesOS =>
            val trackedEntriesOS = new ByteTrackingOutputStream(entriesOS)
            using(makeEntriesEnc(trackedEntriesOS)) { entriesEN =>

              var rowCount = 0L

              val rvb = new RegionValueBuilder()
              val fullRow = new UnsafeRow(fullRowPType)

              it.foreach { rv =>
                fullRow.set(rv)
                val row = fullRow.deleteField(localEntriesIndex)

                val region = rv.region
                rvb.set(region)
                rvb.start(rowsRVType.physicalType)
                rvb.addAnnotation(rowsRVType, row)

                rowsEN.writeByte(1)
                rowsEN.writeRegionValue(region, rvb.end())

                rvb.start(entriesRVType.physicalType)
                rvb.startStruct()
                rvb.addField(fullRowType.physicalType, rv, localEntriesIndex)
                rvb.endStruct()

                entriesEN.writeByte(1)
                entriesEN.writeRegionValue(region, rvb.end())

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
        hConf.copy(rowsPartPath, finalRowsPartPath)
        hConf.copy(entriesPartPath, finalEntriesPartPath)
      }

      Iterator.single(f -> rowCount)
    }.collect()

    val (partFiles, partitionCounts) = partFilePartitionCounts.unzip

    val rowsSpec = AbstractRVDSpec(t.rowType.physicalType, t.rowKey, codecSpec, partFiles, partitioner)
    rowsSpec.write(hConf, path + "/rows/rows")

    val entriesSpec = AbstractRVDSpec(entriesRVType.physicalType, FastIndexedSeq(), codecSpec, partFiles, RVDPartitioner.unkeyed(partitionCounts.length))
    entriesSpec.write(hConf, path + "/entries/rows")

    info(s"wrote ${ partitionCounts.sum } items in $nPartitions partitions to $path")

    partitionCounts
  }

  def toRows(rowType: PStruct): RDD[Row] = {
    crdd.run.map(rv => SafeRow(rowType, rv.region, rv.offset))
  }
}
