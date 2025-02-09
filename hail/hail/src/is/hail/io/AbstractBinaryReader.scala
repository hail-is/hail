package is.hail.io

import java.io._

abstract class AbstractBinaryReader {

  def read(): Int

  def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int

  def readBytes(byteArray: Array[Byte], offset: Int, length: Int): Int = {
    var hasRead = 0
    var toRead = length
    while (toRead > 0) {
      val result = read(byteArray, hasRead, toRead)
      assert(result >= 0)
      hasRead += result
      toRead -= result
    }
    hasRead
  }

  def readBytes(length: Int): Array[Byte] = {
    val arr = new Array[Byte](length)
    readBytes(arr, 0, length)
    arr
  }

  def readLong(): Long =
    (read() & 0xff).asInstanceOf[Long] |
      ((read() & 0xff).asInstanceOf[Long] << 8) |
      ((read() & 0xff).asInstanceOf[Long] << 16) |
      ((read() & 0xff).asInstanceOf[Long] << 24) |
      ((read() & 0xff).asInstanceOf[Long] << 32) |
      ((read() & 0xff).asInstanceOf[Long] << 40) |
      ((read() & 0xff).asInstanceOf[Long] << 48) |
      ((read() & 0xff).asInstanceOf[Long] << 56)

  def readInt(): Int =
    (read() & 0xff) | ((read() & 0xff) << 8) | ((read() & 0xff) << 16) | ((read() & 0xff) << 24)

  def readShort(): Int = (read() & 0xff) | ((read() & 0xff) << 8)

  def readString(length: Int): String = {
    require(length >= 0)
    val byteArray = new Array[Byte](length)
    val result = readBytes(byteArray, 0, length)
    if (result < 0)
      throw new EOFException()

    new String(
      byteArray,
      "iso-8859-1",
    ) // FIXME figure out what BGENs are actually encoding; UTF-8 also works
  }

  def readLengthAndString(lengthBytes: Int): String = {
    require(lengthBytes == 2 || lengthBytes == 4)

    val length = if (lengthBytes == 2) readShort() else readInt()
    readString(length)
  }

  def readLengthAndSkipString(lengthBytes: Int): Unit = {
    require(lengthBytes == 2 || lengthBytes == 4)

    val length = if (lengthBytes == 2) readShort() else readInt()
    skipBytes(length)
  }

  def skipBytes(lengthBytes: Long): Long
}
