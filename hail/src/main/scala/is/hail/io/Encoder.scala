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

final class CompiledEncoder(out: OutputBuffer, f: () => AsmFunction3[Region, Long, OutputBuffer, Unit]) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeRegionValue(region: Region, offset: Long) {
    f()(region, offset, out)
  }

  def writeByte(b: Byte) {
    out.writeByte(b)
  }

  def indexOffset(): Long = out.indexOffset()
}
