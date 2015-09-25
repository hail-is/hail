package org.broadinstitute.hail.io

import org.apache.hadoop.fs.FSDataInputStream

class BetterFSDataInputStream(fis: FSDataInputStream) extends BinaryReader {

  override def read(): Int = {
    fis.read()
  }

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int = {
    fis.read(byteArray, hasRead, toRead)
  }

  override def readBytes(byteArray: Array[Byte], offset: Int, length: Int): Int = {
    fis.read(byteArray, offset, length)
  }

  def close(): Unit = fis.close()

  def seek(pos: Long): Unit = {
    fis.seek(pos)
  }
}