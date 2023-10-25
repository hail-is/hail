package is.hail.io

import java.io._
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.types.encoded.EncoderAsmFunction

trait Encoder extends Closeable {
  def flush(): Unit

  def close(): Unit

  def writeRegionValue(offset: Long): Unit

  def writeByte(b: Byte): Unit

  def indexOffset(): Long
}

final class CompiledEncoder(out: OutputBuffer, ctx: ExecuteContext, f: (ExecuteContext) => EncoderAsmFunction) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  private[this] val compiled = f(ctx)
  def writeRegionValue(offset: Long) {
    compiled(offset, out)
  }

  def writeByte(b: Byte) {
    out.writeByte(b)
  }

  def indexOffset(): Long = out.indexOffset()
}

final class ByteArrayEncoder(
  ctx: ExecuteContext,
  makeEnc: (OutputStream, ExecuteContext) => Encoder
) extends Closeable {
  private[this] val baos = new ByteArrayOutputStream()
  private[this] val enc = makeEnc(baos, ctx)

  def close(): Unit = {
    enc.close()
    baos.close()
  }

  def regionValueToBytes(region: Region, offset: Long): Array[Byte] = {
    reset()
    writeRegionValue(region, offset)
    result()
  }

  def regionValueToBytes(offset: Long): Array[Byte] = {
    baos.reset()
    enc.writeRegionValue(offset)
    result()
  }

  def reset(): Unit = baos.reset()
  def writeRegionValue(region: Region, offset: Long): Unit = enc.writeRegionValue(offset)

  def result(): Array[Byte] = {
    enc.flush()
    baos.toByteArray
  }
}
