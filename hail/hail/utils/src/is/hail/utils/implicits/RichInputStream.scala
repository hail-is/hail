package is.hail.utils.implicits

import is.hail.utils._

import java.io.InputStream

class RichInputStream(val in: InputStream) extends AnyVal {
  def readFully(to: Array[Byte]): Unit =
    readFully(to, 0, to.length)

  def readFully(to: Array[Byte], toOff: Int, n: Int): Unit = {
    val nRead = in.readNBytes(to, toOff, n)
    if (nRead < n) fatal(s"Premature end of file: expected $n bytes, found $nRead")
  }
}
