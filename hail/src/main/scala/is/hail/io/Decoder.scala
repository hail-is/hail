package is.hail.io

import java.io._

import is.hail.annotations.Region
import is.hail.asm4s._

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
