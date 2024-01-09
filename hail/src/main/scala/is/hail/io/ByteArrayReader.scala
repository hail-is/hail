package is.hail.io

class ByteArrayReader(val arr: Array[Byte]) extends AbstractBinaryReader {

  final var position: Int = 0
  val length = arr.length

  override def read(): Int = {
    if (position < length) {
      val b = arr(position)
      position += 1
      b
    } else
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

  def seek(pos: Int) {
    position = pos
  }

  def skipBytes(bytes: Long): Long = {
    require(bytes < Integer.MAX_VALUE)
    position += bytes.toInt
    if (position < length)
      bytes
    else
      -1
  }

  def hasNext(): Boolean = position < length
}
