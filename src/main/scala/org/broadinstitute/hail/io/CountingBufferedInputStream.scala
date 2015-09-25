package org.broadinstitute.hail.io
import java.io._

class CountingBufferedInputStream(bis: BufferedInputStream, start: Long = 0) extends BinaryReader {

  private var position: Long = start

  override def read(): Int = {
    val r = bis.read()
    if (r >= 0)
      position += 1
//    println("read something, pos is %d".format(position))
    r
  }

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int = {
    val bytesRead = bis.read(byteArray, hasRead, toRead)
    position += bytesRead
    bytesRead
  }

  override def readBytes(byteArray: Array[Byte], offset: Int, length: Int): Int = {
    val hasRead = bis.read(byteArray, offset, length)
    position += hasRead
    hasRead
  }

  def getPosition: Long = position
}
