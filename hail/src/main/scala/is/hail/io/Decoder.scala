package is.hail.io

import java.io._

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.utils.RestartableByteArrayInputStream

trait Decoder extends Closeable {
  def close()

  def readRegionValue(region: Region): Long

  def readByte(): Byte

  def seek(offset: Long): Unit
}

final class CompiledDecoder(in: InputBuffer, f: () => AsmFunction2[Region, InputBuffer, Long]) extends Decoder {
  def close() {
    in.close()
  }

  def readByte(): Byte = in.readByte()

  def readRegionValue(region: Region): Long = {
    f()(region, in)
  }

  def seek(offset: Long): Unit = in.seek(offset)
}

final class ByteArrayDecoder(
  makeDec: InputStream => Decoder
) extends Closeable {
  private[this] val bais = new RestartableByteArrayInputStream()
  private[this] val enc = makeDec(bais)

  override def close(): Unit = {
    enc.close()
    bais.close()
  }

  def regionValueFromBytes(region: Region, bytes: Array[Byte]): Long = {
    bais.restart(bytes)
    enc.readRegionValue(region)
  }
}
