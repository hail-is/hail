package is.hail.io

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.encoded.DecoderAsmFunction
import is.hail.types.physical.PType
import is.hail.utils.RestartableByteArrayInputStream

import java.io._

trait Decoder extends Closeable {
  def close()

  def ptype: PType

  def readRegionValue(region: Region): Long

  def readByte(): Byte

  def seek(offset: Long): Unit
}

final class CompiledDecoder(
  in: InputBuffer,
  val ptype: PType,
  theHailClassLoader: HailClassLoader,
  f: (HailClassLoader) => DecoderAsmFunction,
) extends Decoder {
  def close() {
    in.close()
  }

  def readByte(): Byte = in.readByte()

  private[this] val compiled = f(theHailClassLoader)

  def readRegionValue(r: Region): Long =
    compiled(r, in)

  def seek(offset: Long): Unit = in.seek(offset)
}

final class ByteArrayDecoder(
  theHailClassLoader: HailClassLoader,
  makeDec: (InputStream, HailClassLoader) => Decoder,
) extends Closeable {
  private[this] val bais = new RestartableByteArrayInputStream()
  private[this] val dec = makeDec(bais, theHailClassLoader)

  override def close(): Unit = {
    dec.close()
    bais.close()
  }

  def regionValueFromBytes(region: Region, bytes: Array[Byte]): Long = {
    set(bytes)
    readValue(region)
  }

  def readValue(region: Region): Long = dec.readRegionValue(region)

  def set(bytes: Array[Byte]) {
    bais.restart(bytes)
  }
}
