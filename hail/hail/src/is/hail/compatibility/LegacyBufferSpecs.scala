package is.hail.compatibility

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.io._
import is.hail.io.compress.LZ4

import java.io._

final case class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec)
    extends LZ4BlockBufferSpecCommon {
  override def lz4 = LZ4.hc
  override def stagedlz4: Code[LZ4] = Code.invokeScalaObject0[LZ4](LZ4.getClass, "hc")
  override def typeName = "LZ4BlockBufferSpec"
}

final case class LEB128BufferSpec(child: BufferSpec) extends BufferSpec {
  override def buildInputBuffer(in: InputStream): InputBuffer =
    new LEB128InputBuffer(child.buildInputBuffer(in))

  override def buildOutputBuffer(out: OutputStream): OutputBuffer =
    new LEB128OutputBuffer(child.buildOutputBuffer(out))

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer] =
    Code.newInstance[LEB128InputBuffer, InputBuffer](child.buildCodeInputBuffer(in))

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBuffer] =
    Code.newInstance[LEB128OutputBuffer, OutputBuffer](child.buildCodeOutputBuffer(out))
}

final class LEB128InputBuffer(in: InputBuffer) extends InputBuffer {
  override def close(): Unit =
    in.close()

  override def seek(offset: Long): Unit = in.seek(offset)

  override def readByte(): Byte =
    in.readByte()

  override def read(buf: Array[Byte], toOff: Int, n: Int) = in.read(buf, toOff, n)

  override def readInt(): Int = readVarint()

  override def readLong(): Long = readVarintLong()

  override def readFloat(): Float = in.readFloat()

  override def readDouble(): Double = in.readDouble()

  override def readBytes(toRegion: Region, toOff: Long, n: Int): Unit =
    in.readBytes(toRegion, toOff, n)

  override def readBytesArray(n: Int): Array[Byte] = in.readBytesArray(n)

  override def skipByte(): Unit = in.skipByte()

  override def skipInt(): Unit = skipVarint()

  override def skipLong(): Unit = skipVarint()

  override def skipFloat(): Unit = in.skipFloat()

  override def skipDouble(): Unit = in.skipDouble()

  override def skipBytes(n: Int): Unit = in.skipBytes(n)

  override def readDoubles(to: Array[Double], toOff: Int, n: Int): Unit =
    in.readDoubles(to, toOff, n)
}

final class LEB128OutputBuffer(out: OutputBuffer) extends OutputBuffer {
  override def flush(): Unit = out.flush()

  override def close(): Unit =
    out.close()

  override def indexOffset(): Long = out.indexOffset()

  override def writeByte(b: Byte): Unit = out.writeByte(b)

  override def writeInt(i: Int): Unit = writeVarint(i)

  override def writeLong(l: Long): Unit = writeVarintLong(l)

  override def writeFloat(f: Float): Unit = out.writeFloat(f)

  override def writeDouble(d: Double): Unit = out.writeDouble(d)

  override def writeBytes(region: Region, off: Long, n: Int): Unit = out.writeBytes(region, off, n)

  override def writeBytes(addr: Long, n: Int): Unit = out.writeBytes(addr, n)

  override def writeDoubles(from: Array[Double], fromOff: Int, n: Int): Unit =
    out.writeDoubles(from, fromOff, n)
}
