package org.broadinstitute.hail.io

class ByteArrayReader(arr: Array[Byte]) extends BinaryReader {

  def this(bb: ByteBlock) = this(bb.getArray)

  final var position: Int = 0
  val length = arr.length

  override def read(): Int = {
    if (position < length) {
      position += 1
      arr(position-1)
    }
    else
      -1
  }

  override def read(byteArray: Array[Byte], hasRead: Int, toRead: Int): Int = {
    if (position + toRead > length)
      -1
    else {
      System.arraycopy(arr, position, byteArray, hasRead, toRead)
      position += toRead
      toRead
    }
  }

  def seek(pos: Int): Unit = {
    position = pos
  }
}