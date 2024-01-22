package is.hail.io.compress

import java.io.OutputStream
import java.util.zip.{CRC32, Deflater}

import org.apache.hadoop.io.compress.CompressionOutputStream

class BGzipConstants {
  val blockHeaderLength = 18 // Number of bytes in the gzip block before the deflated data.

  val blockLengthOffset =
    16 // Location in the gzip block of the total block size (actually total block size - 1)
  val blockFooterLength = 8 // Number of bytes that follow the deflated data

  val maxCompressedBlockSize =
    64 * 1024 // We require that a compressed block (including header and footer, be <= this)

  val gzipOverhead =
    blockHeaderLength + blockFooterLength + 2 // Gzip overhead is the header, the footer, and the block size (encoded as a short).

  val noCompressionOverhead =
    10 // If Deflater has compression level == NO_COMPRESSION, 10 bytes of overhead (determined experimentally).

  val defaultUncompressedBlockSize =
    64 * 1024 - (gzipOverhead + noCompressionOverhead) // Push out a gzip block when this many uncompressed bytes have been accumulated.

  // gzip magic numbers
  val gzipId1 = 31
  val gzipId2 = 139

  val gzipModificationTime = 0
  val gzipFlag = 4 // set extra fields to true
  val gzipXFL = 0 // extra flags
  val gzipXLEN = 6 // length of extra subfield
  val gzipCMDeflate = 8 // The deflate compression, which is customarily used by gzip
  val defaultCompressionLevel = 5
  val gzipOsUnknown = 255
  val bgzfId1 = 66
  val bgzfId2 = 67
  val bgzfLen = 2

  val emptyGzipBlock = Array(0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
    0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00).map(_.toByte)
}

class BGzipOutputStream(out: OutputStream) extends CompressionOutputStream(out) {
  private[this] var finished: Boolean = false

  val constants = new BGzipConstants
  var numUncompressedBytes = 0
  var uncompressedBuffer = new Array[Byte](constants.defaultUncompressedBlockSize)

  var compressedBuffer =
    new Array[Byte](constants.maxCompressedBlockSize - constants.blockHeaderLength)

  val deflater = new Deflater(constants.defaultCompressionLevel, true)
  val noCompressionDeflater = new Deflater(Deflater.NO_COMPRESSION, true)
  val crc32 = new CRC32

  def write(b: Int): Unit = {
    require(numUncompressedBytes < uncompressedBuffer.length)
    uncompressedBuffer(numUncompressedBytes) = b.toByte
    numUncompressedBytes += 1

    if (numUncompressedBytes == uncompressedBuffer.length)
      deflateBlock()
  }

  override def write(bytes: Array[Byte], offset: Int, length: Int): Unit = {
    require(numUncompressedBytes < uncompressedBuffer.length)

    var currentPosition = offset
    var numBytesRemaining = length

    while (numBytesRemaining > 0) {
      val bytesToWrite =
        math.min(uncompressedBuffer.length - numUncompressedBytes, numBytesRemaining)
      System.arraycopy(bytes, currentPosition, uncompressedBuffer, numUncompressedBytes,
        bytesToWrite)
      numUncompressedBytes += bytesToWrite
      currentPosition += bytesToWrite
      numBytesRemaining -= bytesToWrite
      require(numBytesRemaining >= 0)

      if (numUncompressedBytes == uncompressedBuffer.length)
        deflateBlock()
    }
  }

  final protected def deflateBlock(): Unit = {
    require(numUncompressedBytes != 0)
    assert(!finished)

    deflater.reset()
    deflater.setInput(uncompressedBuffer, 0, numUncompressedBytes)
    deflater.finish()
    var compressedSize: Int = deflater.deflate(compressedBuffer, 0, compressedBuffer.length)

    // If it didn't all fit in compressedBuffer.length, set compression level to NO_COMPRESSION
    // and try again.  This should always fit.
    if (!deflater.finished) {
      noCompressionDeflater.reset()
      noCompressionDeflater.setInput(uncompressedBuffer, 0, numUncompressedBytes)
      noCompressionDeflater.finish()
      compressedSize = noCompressionDeflater.deflate(compressedBuffer, 0, compressedBuffer.length)
      require(noCompressionDeflater.finished)
    }
    // Data compressed small enough, so write it out.
    crc32.reset()
    crc32.update(uncompressedBuffer, 0, numUncompressedBytes)

    writeGzipBlock(compressedSize, numUncompressedBytes, crc32.getValue)

    numUncompressedBytes = 0 // reset variable
  }

  def writeInt8(i: Int) =
    out.write(i & 0xff)

  def writeInt16(i: Int) = {
    out.write(i & 0xff)
    out.write((i >> 8) & 0xff)
  }

  def writeInt32(i: Int) = {
    out.write(i & 0xff)
    out.write((i >> 8) & 0xff)
    out.write((i >> 16) & 0xff)
    out.write((i >> 24) & 0xff)
  }

  def writeGzipBlock(compressedSize: Int, bytesToCompress: Int, crc32val: Long): Int = {
    val totalBlockSize = compressedSize + constants.blockHeaderLength + constants.blockFooterLength

    writeInt8(constants.gzipId1)
    writeInt8(constants.gzipId2)
    writeInt8(constants.gzipCMDeflate)
    writeInt8(constants.gzipFlag)
    writeInt32(constants.gzipModificationTime)
    writeInt8(constants.gzipXFL)
    writeInt8(constants.gzipOsUnknown)
    writeInt16(constants.gzipXLEN)
    writeInt8(constants.bgzfId1)
    writeInt8(constants.bgzfId2)
    writeInt16(constants.bgzfLen)
    writeInt16(totalBlockSize - 1)
    out.write(compressedBuffer, 0, compressedSize)
    writeInt32(crc32val.toInt)
    writeInt32(bytesToCompress)
    totalBlockSize
  }

  def resetState() = throw new UnsupportedOperationException

  override def finish(): Unit = {
    if (numUncompressedBytes != 0)
      deflateBlock()
    if (!finished) {
      out.write(constants.emptyGzipBlock)
      finished = true
    }
  }
}

class ComposableBGzipOutputStream(out: OutputStream) extends BGzipOutputStream(out) {
  override def finish() = if (numUncompressedBytes != 0) {
    deflateBlock()
  }
}
