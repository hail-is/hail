package is.hail.utils

import java.io.InputStream


class SlicedInputStream (
  private[this] val in: InputStream,
  private[this] var n: Long = -1
) extends InputStream {
  private[this] var i = 0
  override def read(): Int =
    if (n == -1 || i < n) {
      i += 1
      in.read()
    } else {
      -1
    }
  override def close(): Unit = in.close()
  def startSlice(_n: Long): Unit = {
    i = 0
    n = _n
  }
}
