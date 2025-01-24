package is.hail.io.compress

import net.jpountz.lz4.{LZ4Compressor, LZ4Factory, LZ4FastDecompressor}

object LZ4 {
  val factory = LZ4Factory.fastestInstance()

  val hc = new LZ4(factory.highCompressor(), factory.fastDecompressor())
  val fast = new LZ4(factory.fastCompressor(), factory.fastDecompressor())
}

class LZ4 private (
  compressor: LZ4Compressor,
  decompressor: LZ4FastDecompressor,
) {
  def maxCompressedLength(decompLen: Int): Int =
    compressor.maxCompressedLength(decompLen)

  def compress(comp: Array[Byte], compOff: Int, decomp: Array[Byte], decompLen: Int): Int = {
    val maxLen = maxCompressedLength(decompLen)
    assert(comp.length >= compOff + maxLen)
    val compressedLen = compressor.compress(decomp, 0, decompLen, comp, compOff, maxLen)
    compressedLen
  }

  def decompress(
    decomp: Array[Byte],
    decompOff: Int,
    decompLen: Int,
    comp: Array[Byte],
    compOff: Int,
    compLen: Int,
  ): Unit = {
    val compLen2 = decompressor.decompress(comp, compOff, decomp, decompOff, decompLen)
    assert(compLen2 == compLen)
  }
}
