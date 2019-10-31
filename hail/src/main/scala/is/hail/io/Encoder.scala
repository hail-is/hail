package is.hail.io

import java.io._

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types.encoded.EncoderAsmFunction

trait Encoder extends Closeable {
  def flush(): Unit

  def close(): Unit

  def writeRegionValue(region: Region, offset: Long): Unit

  def writeByte(b: Byte): Unit

  def indexOffset(): Long
}

final class CompiledEncoder(out: OutputBuffer, f: () => EncoderAsmFunction) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  private[this] val compiled = f()
  def writeRegionValue(region: Region, offset: Long) {
    compiled(offset, out)
  }

  def writeByte(b: Byte) {
    out.writeByte(b)
  }

  def indexOffset(): Long = out.indexOffset()
}

final class ByteArrayEncoder(
  makeEnc: OutputStream => Encoder
) extends Closeable {
  private[this] val baos = new ByteArrayOutputStream()
  private[this] val enc = makeEnc(baos)

  def close(): Unit = {
    enc.close()
    baos.close()
  }

  def regionValueToBytes(region: Region, offset: Long): Array[Byte] = {
    baos.reset()
    enc.writeRegionValue(region, offset)
    enc.flush()
    baos.toByteArray()
  }
}
