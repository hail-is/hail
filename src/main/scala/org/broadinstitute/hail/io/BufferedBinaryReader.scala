package org.broadinstitute.hail.io

import java.io._

class BufferedBinaryReader(bis: BufferedInputStream) extends AbstractBinaryReader {

  def this(is: InputStream) = this(new BufferedInputStream(is))

  def this(path: String) = this(new FileInputStream(path))

  override def read(): Int = bis.read()

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int = bis.read(byteArray, hasRead, toRead)

  override def readBytes(byteArray: Array[Byte], offset: Int, length: Int): Int = bis.read(byteArray, offset, length)

  def skipBytes(bytes: Long): Long = bis.skip(bytes)
}
