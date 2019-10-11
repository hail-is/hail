package is.hail.io

import java.io._

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.utils.RestartableByteArrayInputStream
import is.hail.expr.types.encoded.DecoderAsmFunction

trait Decoder extends Closeable {
  def close()

  def readRegionValue(region: Region): Long

  def readByte(): Byte

  def seek(offset: Long): Unit
}

final class CompiledDecoder(in: InputBuffer, f: () => DecoderAsmFunction) extends Decoder {
  def close() {
    in.close()
  }

  def readByte(): Byte = in.readByte()

  private[this] val compiled = f()
  def readRegionValue(r: Region): Long = {
    compiled(r, in)
  }

  def seek(offset: Long): Unit = in.seek(offset)
}

final class ByteArrayDecoder(
  makeDec: InputStream => Decoder
) extends Closeable {
  private[this] val bais = new RestartableByteArrayInputStream()
  private[this] val dec = makeDec(bais)

  override def close(): Unit = {
    dec.close()
    bais.close()
  }

  def regionValueFromBytes(region: Region, bytes: Array[Byte]): Long = {
    bais.restart(bytes)
    dec.readRegionValue(region)
  }
}
