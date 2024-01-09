package is.hail.io

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.encoded.EncoderAsmFunction

import java.io._

trait Encoder extends Closeable {
  def flush(): Unit

  def close(): Unit

  def writeRegionValue(offset: Long): Unit

  def writeByte(b: Byte): Unit

  def indexOffset(): Long
}

final class CompiledEncoder(
  out: OutputBuffer,
  theHailClassLoader: HailClassLoader,
  f: (HailClassLoader) => EncoderAsmFunction,
) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  private[this] val compiled = f(theHailClassLoader)

  def writeRegionValue(offset: Long) {
    compiled(offset, out)
  }

  def writeByte(b: Byte) {
    out.writeByte(b)
  }

  def indexOffset(): Long = out.indexOffset()
}

final class ByteArrayEncoder(
  theHailClassLoader: HailClassLoader,
  makeEnc: (OutputStream, HailClassLoader) => Encoder,
) extends Closeable {
  private[this] val baos = new ByteArrayOutputStream()
  private[this] val enc = makeEnc(baos, theHailClassLoader)

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
