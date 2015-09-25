package org.broadinstitute.hail.io

import java.io._

class BinaryFileReader(raf: RandomAccessFile) extends BinaryReader {

  def this(path: String) = {
    this(new RandomAccessFile(path, "r"))
  }

  private var cbis: CountingBufferedInputStream = new CountingBufferedInputStream(new BufferedInputStream(new
    FileInputStream(raf.getFD)))

  private def uIntToLong(i: Int): Long = i.toLong & 0xffffffff

  private def uShortToInt(i: Short): Int = i.toInt & 0xffff

  def seek(pos: Long) = {
    raf.seek(pos)
    cbis = new CountingBufferedInputStream(new BufferedInputStream(new
        FileInputStream(raf.getFD)), start=pos)
  }

  override def read(): Int = cbis.read()

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int = cbis.read(byteArray, hasRead, toRead)

  override def readBytes(length: Int) = {
    cbis.readBytes(length)
  }

  def getPosition: Long = cbis.getPosition


}