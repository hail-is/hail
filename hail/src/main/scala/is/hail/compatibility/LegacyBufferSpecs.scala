package is.hail.compatibility

import is.hail.io.compress.LZ4
import is.hail.io._

@deprecated("LZ4HCBlockBufferSpec is a drop-in replacement for this class", "Hail 0.2.24")
final case class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec)
    extends LZ4BlockBufferSpecCommon {
  def lz4 = LZ4.hc
  def typeName = "LZ4BlockBufferSpec"
}

