package is.hail.io

import java.io._

import is.hail.annotations.Region
import is.hail.asm4s._

trait Encoder extends Closeable {
  def flush(): Unit

  def close(): Unit

  def writeRegionValue(region: Region, offset: Long): Unit

  def writeByte(b: Byte): Unit

  def indexOffset(): Long
}

final class CompiledEncoder(out: OutputBuffer, f: () => AsmFunction2[Long, OutputBuffer, Unit]) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeRegionValue(region: Region, offset: Long) {
    f()(offset, out)
  }

  def writeByte(b: Byte) {
    out.writeByte(b)
  }

  def indexOffset(): Long = out.indexOffset()
}

final class ByteArrayEncoder(
  makeEnc: OutputStream => Encoder
) extends Encoder {
  private[this] val baos = new ByteArrayOutputStream()
  private[this] val enc = makeEnc(baos)

  def flush(): Unit = enc.flush()

  def close(): Unit = {
    enc.close()
    baos.close()
  }

  def writeRegionValue(region: Region, offset: Long): Unit =
    enc.writeRegionValue(region, offset)

  def writeByte(b: Byte): Unit =
    enc.writeByte(b)

  def indexOffset(): Long =
    enc.indexOffset()

  def toByteArray(): Array[Byte] =
    baos.toByteArray()

  def restart() = baos.reset()

  def regionValueToBytes(region: Region, offset: Long): Array[Byte] = {
    restart()
    writeRegionValue(region, offset)
    flush()
    toByteArray()
  }
}
