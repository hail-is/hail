package is.hail.utils

import is.hail.io.fs.Seekable

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

  def seek(offset: Long): Unit = {
    base match {
      case base: Seekable =>
        base.seek(offset)
      case base: org.apache.hadoop.fs.Seekable =>
        base.seek(offset)
    }
  }
}
