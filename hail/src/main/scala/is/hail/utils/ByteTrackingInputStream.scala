package is.hail.utils

import java.io.InputStream

class ByteTrackingInputStream(base: InputStream) extends InputStream {
  var bytesRead = 0L

  def bytesReadAndClear(): Long = {
    val n = bytesRead
    bytesRead = 0L
    n
  }

  override def read(): Int = {
    bytesRead += 1
    base.read()
  }

  override def read(b: Array[Byte]): Int = {
    val n = base.read(b)
    bytesRead += n
    n
  }

  override def read(b: Array[Byte], off: Int, len: Int): Int = {
    val n = base.read(b, off, len)
    bytesRead += n
    n
  }

  override def close(): Unit = base.close()
}
