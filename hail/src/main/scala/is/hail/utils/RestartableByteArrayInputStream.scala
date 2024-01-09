package is.hail.utils

import java.io.{InputStream, IOException}

// not thread safe
class RestartableByteArrayInputStream extends InputStream {
  private[this] var off: Int = 0
  private[this] var end: Int = 0
  private[this] var buf: Array[Byte] = null

  def this(buf: Array[Byte]) {
    this()
    restart(buf)
  }

  override def read(): Int = {
    if (off == end) {
      return -1
    }
    val b = buf(off) & 0xff
    off += 1
    b
  }

  override def read(dest: Array[Byte]): Int =
    read(dest, 0, dest.length)

  override def read(dest: Array[Byte], destOff: Int, requestedLength: Int): Int = {
    val length = math.min(requestedLength, end - off)
    System.arraycopy(buf, off, dest, destOff, length)
    off += length
    length
  }

  override def skip(n: Long): Long = {
    if (n <= 0) {
      return 0
    }
    val skipped = math.min(
      math.min(n, Integer.MAX_VALUE).toInt,
      end - off,
    )
    off += skipped
    skipped
  }

  override def available(): Int =
    end - off

  override def markSupported(): Boolean = false

  override def mark(readAheadLimit: Int): Unit =
    throw new IOException("unsupported operation")

  override def reset(): Unit =
    throw new IOException("unsupported operation")

  override def close(): Unit =
    buf = null

  def restart(buf: Array[Byte]): Unit =
    restart(buf, 0, buf.length)

  def restart(buf: Array[Byte], start: Int, end: Int): Unit = {
    require(start >= 0)
    require(start <= end)
    require(end <= buf.length)
    this.buf = buf
    this.off = start
    this.end = end
  }
}
