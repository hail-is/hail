package is.hail.io.fs

import java.io._

class DoubleCloseSafeOutputStream (
  private[this] val out: OutputStream
) extends OutputStream {
  private[this] var closed: Boolean = False
  def write(b: Int) { out.write(b) }
  def write(b: Array[Byte]) { out.write(b) }
  def write(b: Array[Byte], off: Int, len: Int) { out.write(b, off, len) }
  def flush() { out.flush() }
  def close() = this.synchronized {
    if (!closed) {
      out.close()
      closed = true
    }
  }
}
