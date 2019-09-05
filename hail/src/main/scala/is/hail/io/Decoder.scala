package is.hail.io

import java.io._

import is.hail.annotations.Region

trait Decoder extends Closeable {
  def close()

  def readRegionValue(region: Region): Long

  def readByte(): Byte

  def seek(offset: Long): Unit
}
