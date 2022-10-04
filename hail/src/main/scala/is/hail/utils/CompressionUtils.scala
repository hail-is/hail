package is.hail.utils

import is.hail.expr.ir.ByteArrayBuilder

import java.util.zip.{Deflater, Inflater}

object CompressionUtils {
  def compress(bb: ByteArrayBuilder, input: Array[Byte]): Int = {
    val compressor = new Deflater()
    compressor.setInput(input)
    compressor.finish()
    val buffer = new Array[Byte](1024)
    var compressedLength = 0
    while (!compressor.finished()) {
      val nCompressedBytes = compressor.deflate(buffer)
      bb ++= (buffer, nCompressedBytes)
      compressedLength += nCompressedBytes
    }
    compressedLength
  }

  def decompress(input: Array[Byte], size: Int): Array[Byte] = {
    val expansion = new Array[Byte](size)
    val inflater = new Inflater
    inflater.setInput(input)
    var off = 0
    while (off < expansion.length) {
      off += inflater.inflate(expansion, off, expansion.length - off)
    }
    expansion
  }
}
