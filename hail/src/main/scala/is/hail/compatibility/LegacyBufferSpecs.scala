package is.hail.compatibility

import is.hail.io.compress.LZ4
import is.hail.io._

final case class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec)
    extends LZ4BlockBufferSpecCommon {
  def lz4 = LZ4.hc
  def typeName = "LZ4BlockBufferSpec"
}

