package is.hail.variant

import java.util

import net.jpountz.lz4.LZ4Factory

object LZ4Utils {
  val factory = LZ4Factory.fastestInstance()
  val compressor = factory.highCompressor()
  val decompressor = factory.fastDecompressor()

  def maxCompressedLength(decompLen: Int): Int =
    compressor.maxCompressedLength(decompLen)

  def compress(comp: Array[Byte], decomp: Array[Byte], decompLen: Int): Int = {
    val maxLen = maxCompressedLength(decompLen)
    assert(comp.length >= maxLen)
    val compressedLen = compressor.compress(decomp, 0, decompLen, comp, 0, maxLen)
    compressedLen
  }

  def decompress(decomp: Array[Byte], decompLen: Int, comp: Array[Byte], compLen: Int) {
    val compLen2 = decompressor.decompress(comp, 0, decomp, 0, decompLen)
    assert(compLen2 == compLen)
  }
}
