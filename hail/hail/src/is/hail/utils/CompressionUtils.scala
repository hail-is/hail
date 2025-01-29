package is.hail.utils

import is.hail.expr.ir.ByteArrayBuilder

import java.util.zip.{Deflater, Inflater}

import com.github.luben.zstd.Zstd

object CompressionUtils {
  def compressZlib(bb: ByteArrayBuilder, input: Array[Byte]): Int = {
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

  def compressZstd(bb: ByteArrayBuilder, input: Array[Byte]): Int = {
    val maxSize = Zstd.compressBound(input.length).toInt
    val sizeBefore = bb.size
    bb.ensureCapacity(bb.size + maxSize)
    val compressedSize =
      Zstd.compressByteArray(bb.b, sizeBefore, maxSize, input, 0, input.length, 5).toInt
    bb.setSizeUnchecked(sizeBefore + compressedSize)
    compressedSize
  }

  def decompressZlib(input: Array[Byte], size: Int): Array[Byte] = {
    val expansion = new Array[Byte](size)
    val inflater = new Inflater
    inflater.setInput(input)
    var off = 0
    while (off < expansion.length)
      off += inflater.inflate(expansion, off, expansion.length - off)
    expansion
  }

  def decompressZstd(input: Array[Byte], size: Int): Array[Byte] = {
    val expansion = new Array[Byte](size)
    val n = Zstd.decompress(expansion, input)
    assert(n == size)
    expansion
  }
}
