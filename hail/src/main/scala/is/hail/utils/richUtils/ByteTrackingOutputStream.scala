package is.hail.utils.richUtils

import java.io.OutputStream

class ByteTrackingOutputStream(base: OutputStream) extends OutputStream {
  var bytesWritten = 0L

  def write(c: Int) {
    bytesWritten += 1
    base.write(c)
  }

  override def write(b: Array[Byte]) {
    base.write(b)
    bytesWritten += b.length
  }

  override def write(b: Array[Byte], off: Int, len: Int) {
    base.write(b, off, len)
    bytesWritten += len
  }

  override def close(): Unit = base.close()
}
