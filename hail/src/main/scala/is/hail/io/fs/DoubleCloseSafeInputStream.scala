package is.hail.io.fs

import java.io._

class DoubleCloseSafeInputStream (
  private[this] val in: InputStream
) extends InputStream {
  private[this] var closed: Boolean = False
  def read(): Int = in.read()
  def read(b: Array[Byte]): Int = in.read(b)
  def read(b: Array[Byte], off: Int, len: Int): Int = in.read(b, off, len)
  def skip(n: Long): Long = in.skip(n)
  def available(): Int = in.available()
  def mark(readlimit: Int) { in.mark(readlimit) }
  def reset() { in.reset() }
  def markSupported(): Boolean = in.markSupported()
  def close() = this.synchronized {
    if (!closed) {
      in.close()
      closed = true
    }
  }
}
