package is.hail.io

import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.io.compress.LZ4Utils
import is.hail.nativecode._
import is.hail.rvd.{OrderedRVDSpec, RVDContext, RVDPartitioner, RVDSpec, UnpartitionedRVDSpec}
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.json4s.{Extraction, JValue}
import org.json4s.jackson.JsonMethods
import java.io.{Closeable, InputStream, OutputStream, PrintWriter}

import scala.collection.mutable.ArrayBuffer
import is.hail.asm4s._
import is.hail.expr.ir.{EmitUtils, EstimableEmitter, MethodBuilderLike}
import is.hail.expr.types.MatrixType
import is.hail.expr.types.physical._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.apache.spark.sql.Row
import org.apache.spark.{ExposedMetrics, TaskContext}

trait BufferSpec extends Serializable {
  def buildInputBuffer(in: InputStream): InputBuffer

  def buildOutputBuffer(out: OutputStream): OutputBuffer
}

final class LEB128BufferSpec(child: BufferSpec) extends BufferSpec {
  def buildInputBuffer(in: InputStream): InputBuffer = new LEB128InputBuffer(child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new LEB128OutputBuffer(child.buildOutputBuffer(out))
}

final class BlockingBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BufferSpec {
  def buildInputBuffer(in: InputStream): InputBuffer = new BlockingInputBuffer(blockSize, child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new BlockingOutputBuffer(blockSize, child.buildOutputBuffer(out))
}

trait BlockBufferSpec extends Serializable {
  def buildInputBuffer(in: InputStream): InputBlockBuffer

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer
}

final class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BlockBufferSpec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer = new LZ4InputBlockBuffer(blockSize, child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer = new LZ4OutputBlockBuffer(blockSize, child.buildOutputBuffer(out))
}

object StreamBlockBufferSpec {
  def extract(jv: JValue): StreamBlockBufferSpec = new StreamBlockBufferSpec
}

final class StreamBlockBufferSpec extends BlockBufferSpec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer = new StreamBlockInputBuffer(in)

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer = new StreamBlockOutputBuffer(out)
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
    implicit val formats = RVDSpec.formats
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
      val x = (buf(pos+j).toInt & 0xff)
      if (x <= 0xf) sb.append(s" 0${x.toHexString}") else sb.append(s" ${x.toHexString}")
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
      val x = (Memory.loadByte(addr+j).toInt & 0xff)
      if (x <= 0xf) sb.append(s" 0${x.toHexString}") else sb.append(s" ${x.toHexString}")
      if ((j & 0x7) == 0x7) sb.append("\n")
      j += 1
    }
    System.err.println(sb.toString())
  }

}

final case class PackCodecSpec(child: BufferSpec) extends CodecSpec {

  def buildEncoder(t: PType): (OutputStream) => Encoder = { out: OutputStream =>
    new PackEncoder(t, child.buildOutputBuffer(out))
  }
  
  def buildDecoder(t: PType, requestedType: PType): (InputStream) => Decoder = {
    if (System.getenv("HAIL_ENABLE_CPP_CODEGEN") != null) {
      val sb = new StringBuilder()
      NativeDecode.appendCode(sb, t, requestedType)
      val code = new PrettyCode(sb.toString())
      // Experiments on Mac (LLVM) show -O2 code about 1.08x faster than -O1,
      // but -O3 is no better than -O2.
      // UnsafeSuite.testCodec takes a long time on Mac with -O2, so use -O1
      val options = "-O1"
      val mod = new NativeModule(options, code.toString())
      val st = new NativeStatus()
      mod.findOrBuild(st)
      assert(st.ok, st.toString())
      st.close()
      val modKey = mod.getKey
      val modBinary = mod.getBinary
      mod.close()
      (in: InputStream) => new NativePackDecoder(child.buildInputBuffer(in), modKey, modBinary)
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

  // C++ decoder must buffer data ahead of the decode to go fast, but must not
  // go across a block boundary because IndexReader will seek() the InputStream
  // after each RegionValue, invalidating any read-ahead data.
  def readToEndOfBlock(toAddr: Long, toBuf: Array[Byte], toOff: Int, n: Int): Int
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

  def readToEndOfBlock(toAddr: Long, toBuf: Array[Byte], toOff: Int, n: Int): Int = {
    val result = in.readToEndOfBlock(toAddr, toBuf, toOff, n)
    assert(result <= n)
    result
  }
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

  def readToEndOfBlock(toAddr: Long, toBuf: Array[Byte], toOff: Int, n: Int): Int = {
    var ngot = (lim - pos)
    while (ngot == 0) {
      pos = 0
      ngot = readBlock(decompBuf)
    }
    if (ngot > 0) {
      if (ngot > n) ngot = n
      if (toAddr != 0) { // copy directly to off-heap buffer
        Memory.memcpy(toAddr+toOff, decompBuf, pos, ngot)
      } else {
        Memory.memcpy(toBuf, toOff, decompBuf, pos, ngot)
      }
      pos += ngot
    }
    assert(ngot <= n)
    ngot
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

  def readToEndOfBlock(toAddr: Long, toBuf: Array[Byte], toOff: Int, n: Int): Int = {
    var ngot = (end - off)
    while (ngot == 0) {
      readBlock()
      ngot = end
    }
    if (ngot > 0) {
      if (ngot > n) ngot = n
      if (toAddr != 0) { // copy directly to off-heap buffer
        Memory.memcpy(toAddr+toOff, buf, off, ngot)
      } else {
        Memory.memcpy(toBuf, toOff, buf, off, ngot)
      }
      off += ngot
    }
    assert(ngot <= n)
    ngot
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

//
// Generate the Type-specific C++ code for a PackDecoder
//
object NativeDecode {

  def appendCode(sb: StringBuilder, rowType: PType, wantType: PType): Unit = {
    val verbose = false
    var seen = new ArrayBuffer[Int]()
    val localDefs = new StringBuilder()
    val entryCode = new StringBuilder()
    val mainCode = new StringBuilder()

    def stateVarType(name: String): String = {
      name match {
        case "len" => "ssize_t"
        case "idx" => "ssize_t"
        case "miss" => "std::vector<char>"
        case _ => "char*"
      }
    }

    def stateVar(name: String, depth: Int): String = {
      val bit = name match {
        case "len"  => 0x01
        case "idx"  => 0x02
        case "addr" => 0x04
        case "ptr"  => 0x08
        case "data" => 0x10
        case "miss" => 0x20
      }
      if (seen.length <= depth) seen = seen.padTo(depth+1, 0)
      val result = s"${name}${depth}"
      if ((seen(depth) & bit) == 0) {
        seen(depth) = (seen(depth) | bit)
        val typ = stateVarType(name)
        val initStr =
          if (typ.equals("std::vector<char>")) ""
          else if (!typ.equals("char*")) " = 0"
          else " = nullptr"
        localDefs.append(s"${typ} ${result}${initStr};\n")
      }
      result
    }

    var numStates = 0
    def allocState(name: String): Int = {
      val s = numStates
      numStates += 1
      entryCode.append(s"      case ${s}: goto entry${s};\n")
      mainCode.append(s"    entry${s}: // ${name}\n")
      if (verbose) mainCode.append(s"""    fprintf(stderr, "DEBUG: %p entry${s} ${name}\\n", this);\n""")
      s
    }

    def isEntryPoint(t: PType): Boolean = {
      t match {
        case _: PBaseStruct => false
        case _ => true
      }
    }

    def isEmptyStruct(t: PType): Boolean = {
      // A struct which no fields, except other empty structs
      if (t.byteSize == 0) true else false
    }

    def scan(depth: Int, name: String, typ: PType, wantType: PType, skip: Boolean, inBaseAddr: String, off: Long) {
      val r1 = if (isEntryPoint(typ)) allocState(name) else -1
      var baseAddr = inBaseAddr
      var addr = if (off == 0) baseAddr else s"(${baseAddr}+${off})"
      typ.fundamentalType match {
        case t: PBoolean =>
          val call = if (skip) "this->skip_byte()" else s"this->decode_byte((int8_t*)${addr})"
          mainCode.append(s"if (!${call}) { s = ${r1}; goto pull; }\n")
        case t: PInt32 =>
          val call = if (skip) "this->skip_int()" else s"this->decode_int((int32_t*)${addr})"
          mainCode.append(s"if (!${call}) { s = ${r1}; goto pull; }\n")
        case t: PInt64 =>
          val call = if (skip) "this->skip_long()" else s"this->decode_long((int64_t*)${addr})"
          mainCode.append(s"if (!${call}) { s = ${r1}; goto pull; }\n")
        case t: PFloat32 =>
          val call = if (skip) "this->skip_float()" else s"this->decode_float((float*)${addr})"
          mainCode.append(s"if (!${call}) { s = ${r1}; goto pull; }\n")
        case t: PFloat64 =>
          val call = if (skip) "this->skip_double()" else s"this->decode_double((double*)${addr})"
          mainCode.append(s"if (!${call}) { s = ${r1}; goto pull; }\n")

        case t: PBinary =>
          // TBinary - usually a string - has an int length, followed by that number of bytes
          val ptr = stateVar("ptr", depth)
          val len = stateVar("len", depth)
          val idx = stateVar("idx", depth)
          mainCode.append(s"if (!this->decode_length(&${len})) { s = ${r1}; goto pull; }\n")
          if (skip) {
            mainCode.append(s"for (${idx} = 0; ${idx} < ${len};) {\n")
            val r2 = allocState(s"${name}.bytes");
            mainCode.append(s"  auto ngot = this->skip_bytes(${len}-${idx});\n")
            mainCode.append(s"  if (ngot <= 0) { s = ${r2}; goto pull; }\n")
            mainCode.append(s"  ${idx} += ngot;\n")
            mainCode.append(s"}\n")
          } else {
            mainCode.append(s"${ptr} = region->allocate(4, 4+${len});\n")
            mainCode.append(s"*(char**)${addr} = ${ptr};\n")
            mainCode.append(s"*(int32_t*)${ptr} = ${len};\n")
            mainCode.append(s"for (${idx} = 0; ${idx} < ${len};) {\n")
            val r2 = allocState(s"${name}.bytes");
            mainCode.append(s"  auto ngot = this->decode_bytes(${ptr}+4+${idx}, ${len}-${idx});\n")
            mainCode.append(s"  if (ngot <= 0) { s = ${r2}; goto pull; }\n")
            mainCode.append(s"  ${idx} += ngot;\n")
            mainCode.append(s"}\n")
          }

        case t: PArray =>
          val len = stateVar("len", depth)
          val idx = stateVar("idx", depth)
          val ptr = stateVar("ptr", depth)
          val data = if (skip) "data_undefined" else stateVar("data", depth)
          var miss = if (t.elementType.required || !skip) "miss_undefined" else stateVar("miss", depth)
          mainCode.append(s"if (!this->decode_length(&${len})) { s = ${r1}; goto pull; }\n")
          val wantArray = wantType.asInstanceOf[PArray]
          val ealign = wantArray.elementType.alignment
          val align = if (ealign > 4) ealign else 4
          val esize = wantArray.elementByteSize
          val req = if (t.elementType.required) "true" else "false"
          if (skip) {
            if (!t.elementType.required) {
              mainCode.append(s"stretch_size(${miss}, missing_bytes(${len}));\n")
            }
          } else {
            mainCode.append(s"{ ssize_t data_offset = elements_offset(${len}, ${req}, ${ealign});\n")
            mainCode.append(s"  ssize_t size = data_offset + ${esize}*${len};\n")
            mainCode.append(s"  ${ptr} = region->allocate(${align}, size);\n");
            mainCode.append(s"  memset(${ptr}, 0xff, size); // initialize all-missing\n")
            mainCode.append(s"  *(char**)${addr} = ${ptr};\n")
            mainCode.append(s"  ${data} = ${ptr} + data_offset;\n")
            mainCode.append(s"}\n")
            mainCode.append(s"*(int32_t*)${ptr} = ${len};\n")
            miss = s"(${ptr}+4)"
          }
          if (!t.elementType.required) {
            mainCode.append(s"for (${idx} = 0; ${idx} < missing_bytes(${len});) {\n")
            val r2 = allocState(s"${name}.missing");
            mainCode.append(s"  auto ngot = this->decode_bytes(&${miss}[${idx}], missing_bytes(${len})-${idx});\n")
            mainCode.append(s"  if (ngot <= 0) { s = ${r2}; goto pull; }\n")
            mainCode.append(s"  ${idx} += ngot;\n")
            mainCode.append(s"}\n")
          }
          mainCode.append(  s"for (${idx} = 0; ${idx} < ${len}; ++${idx}) {\n")
          if (!t.elementType.required) {
            mainCode.append(s"  if (is_missing(${miss}, ${idx})) continue;\n")
          }
          var elementAddr = "unknown_addr"
          if (!skip && !isEmptyStruct(t.elementType)) {
            elementAddr = stateVar("addr", depth+1)
            mainCode.append(s"  ${elementAddr} = ${data} + ${idx}*${esize};\n")
          }
          scan(depth+1, s"${name}(${idx})", t.elementType, wantArray.elementType, skip, elementAddr, 0)
          mainCode.append(  s"}\n")

        case t: PBaseStruct =>
          val wantStruct = wantType.fundamentalType.asInstanceOf[PBaseStruct];
          var miss = "miss_undefined"
          var shuffleMissingBits = false
          var fieldToWantIdx = new Array[Int](t.fields.length)
          if ((t.nMissingBytes > 0) &&
            (skip || (wantStruct.fields.length < t.fields.length))) {
            miss = stateVar("miss", depth)
            mainCode.append(s"stretch_size(${miss}, ${t.nMissingBytes});\n")
          }
          if (!skip) {
            if (depth == 0) { // top-level PBaseStruct must be allocated
              addr = stateVar("addr", depth)
              baseAddr = addr
              mainCode.append(s"${addr} = region->allocate(${wantStruct.alignment}, ${wantStruct.byteSize});\n")
              if (wantStruct.byteSize > 0) {
                mainCode.append(s"memset(${addr}, 0xff, ${wantStruct.byteSize}); // initialize all-missing\n")
              }
              mainCode.append(s"this->rv_base_ = ${addr};\n")
            }
            var wantIdx = 0
            var fieldIdx = 0
            while (fieldIdx < t.fields.length) {
              val wantName = if (wantIdx < wantStruct.fields.length) wantStruct.fields(wantIdx).name else "~Bad Name~"
              if (t.fields(fieldIdx).name.equals(wantName)) {
                fieldToWantIdx(fieldIdx) = wantIdx
                wantIdx += 1
              } else {
                fieldToWantIdx(fieldIdx) = -1
                shuffleMissingBits = true
              }
              fieldIdx += 1
            }
            var maxMissingBit = -1
            var j = 0
            while (j < wantStruct.missingIdx.length) {
              val bit = wantStruct.missingIdx(j)
              if (maxMissingBit < bit) maxMissingBit = bit
              j += 1
            }
            if (shuffleMissingBits) {
              if (maxMissingBit >= 0) {
                mainCode.append(s"set_all_missing(${addr}, ${maxMissingBit+1});\n")
              }
            } else {
              miss = addr
            }
          }
          if (t.nMissingBytes == 1) {
            val r2 = allocState(s"${name}.missing");
            mainCode.append(s"if (this->decode_bytes(&${miss}[0], 1) <= 0) { s = ${r2}; goto pull; }\n")
          } else if (t.nMissingBytes > 1) {
            // Ack! We have to read this missing bytes, but shuffle bits needed for wantStruct
            val idx = stateVar("idx", depth)
            mainCode.append(s"for (${idx} = 0; ${idx} < ${t.nMissingBytes};) {\n")
            val r2 = allocState(s"${name}.missing")
            mainCode.append(s"  auto ngot = this->decode_bytes(&${miss}[${idx}], ${t.nMissingBytes}-${idx});\n")
            mainCode.append(s"  if (ngot <= 0) { s = ${r2}; goto pull; }\n")
            mainCode.append(s"  ${idx} += ngot;\n")
            mainCode.append(s"}\n")
          }
          var fieldIdx = 0
          while (fieldIdx < t.fields.length) {
            val field = t.fields(fieldIdx)
            val wantIdx = fieldToWantIdx(fieldIdx)
            val fieldSkip = skip || (wantIdx < 0)
            val fieldType = t.types(fieldIdx)
            val wantType = if (fieldSkip) fieldType else wantStruct.types(wantIdx)
            val wantOffset = if (fieldSkip) -1 else wantStruct.byteOffsets(wantIdx)
            if (!t.fieldRequired(fieldIdx)) {
              val m = t.missingIdx(fieldIdx)
              mainCode.append(s"if (!is_missing(${miss}, ${m})) {\n")
              if (!fieldSkip) {
                if (shuffleMissingBits) {
                  val mbit = wantStruct.missingIdx(wantIdx)
                  mainCode.append(s"  ${addr}[${mbit>>3}] &= ~(1<<${mbit&0x7});\n")
                }
              }
              mainCode.append(s"  // ${name}.${field.name} fieldSkip ${fieldSkip} ${fieldType}\n")
              scan(depth+1, s"${name}.${field.name}", fieldType, wantType, fieldSkip, baseAddr, off+wantOffset)
              mainCode.append(s"}\n")
            } else {
              scan(depth+1, s"${name}.${field.name}", fieldType, wantType, fieldSkip, baseAddr, off+wantOffset)
            }
            fieldIdx += 1
          }

        case _ =>
          mainCode.append(s"// unknown type ${typ}\n")
          assert(false)

      }
    }

    allocState("init")
    scan(0, "root", rowType, wantType, false, "unknown_addr", 0)

    sb.append(s"""#include "hail/hail.h"
      |#include "hail/PackDecoder.h"
      |#include "hail/NativeStatus.h"
      |#include "hail/Region.h"
      |#include <cstdint>
      |#include <cstring>
      |#include <cstdio>
      |#include <sys/time.h>
      |
      |NAMESPACE_HAIL_MODULE_BEGIN
      |
      |template<int DecoderId>
      |class Decoder : public PackDecoderBase<DecoderId> {
      | public:
      |  Decoder(ObjectArray* inputArray) {
      |    this->set_input(inputArray);
      |  }
      |
      |  virtual ~Decoder() { }
      |
      |  virtual int64_t decode_one_item(Region* region) {
      |${localDefs}
      |    int s = 0;
      |    for (;;) {
      |      switch (s) {
      |${entryCode}
      |        default: break;
      |      }
      |${mainCode}
      |      return (int64_t)this->rv_base_;
      |""".stripMargin)
      if (rowType.byteSize > 0) {
        sb.append("pull:\n")
        sb.append("  if (this->read_to_end_of_block() < 0) return -1;\n")
      }
      sb.append(s"""
      |    }
      |  }
      |};
      |
      |NativeObjPtr make_decoder(NativeStatus*, long input, long decoderId) {
      |  auto inputArray = reinterpret_cast<ObjectArray*>(input);
      |  if (decoderId == 0) return std::make_shared< Decoder<0> >(inputArray);
      |  if (decoderId == 1) return std::make_shared< Decoder<1> >(inputArray);
      |  return NativeObjPtr();
      |}
      |
      |int64_t decode_one_item(NativeStatus*, long decoder, long region) {
      |  auto obj = (DecoderBase*)decoder;
      |  return obj->decode_one_item((Region*)region);
      |}
      |
      |int64_t decode_one_byte(NativeStatus*, long decoder) {
      |  auto obj = (DecoderBase*)decoder;
      |  return obj->decode_one_byte();
      |}
      |
      |NAMESPACE_HAIL_MODULE_END
      |""".stripMargin)
  }
}

final class NativePackDecoder(in: InputBuffer, moduleKey: String, moduleBinary: Array[Byte]) extends Decoder {
  val st = new NativeStatus()
  val mod = new NativeModule(moduleKey, moduleBinary)
  val make_decoder = mod.findPtrFuncL2(st, "make_decoder")
  assert(st.ok, st.toString())
  val decode_one_byte = mod.findLongFuncL1(st, "decode_one_byte")
  assert(st.ok, st.toString())
  val decode_one_item = mod.findLongFuncL2(st, "decode_one_item")
  assert(st.ok, st.toString())
  val input = new ObjectArray(in)
  val decoder = new NativePtr(make_decoder, st, input.get(), in.decoderId)
  input.close()
  assert(st.ok, st.toString())

  def close(): Unit = {
    decoder.close()
    decode_one_item.close()
    decode_one_byte.close()
    make_decoder.close()
    // NativePtr's to objects with destructors using the module code must
    // *not* be close'd last, since the module will be dlclose'd before the
    // destructor is called.  One safe policy is to close everything in
    // reverse order, ending with the NativeModule
    mod.close()
    st.close()
    in.close()
  }

  def readByte(): Byte = {
    var rc = decode_one_byte(st, decoder.get())
    if (rc < 0) rc = 0
    rc.toByte
  }

  def readRegionValue(region: Region): Long = {
    val result = decode_one_item(st, decoder.get(), region.get())
    if (result == -1L) {
      throw new java.util.NoSuchElementException("NativePackDecoder bad RegionValue")
    }
    result
  }
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

    val rowsSpec = OrderedRVDSpec(t.rowRVDType,
      codecSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(partitioner.rangeBounds, partitioner.rangeBoundsType))
    rowsSpec.write(hConf, path + "/rows/rows")

    val entriesSpec = UnpartitionedRVDSpec(entriesRVType, codecSpec, partFiles)
    entriesSpec.write(hConf, path + "/entries/rows")

    info(s"wrote ${ partitionCounts.sum } items in $nPartitions partitions to $path")

    partitionCounts
  }

  def toRows(rowType: PStruct): RDD[Row] = {
    crdd.run.map(rv => SafeRow(rowType, rv.region, rv.offset))
  }
}
