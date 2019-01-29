package is.hail.utils.richUtils

import java.io.InputStream
import is.hail.utils._

class RichInputStream(val in: InputStream) extends AnyVal {
  def readFully(to: Array[Byte], toOff: Int, n: Int): Unit =
    if (readRepeatedly(to, toOff, n) < n) fatal("Premature end of file")

  def readRepeatedly(to: Array[Byte], toOff: Int, n: Int): Int = {
    assert(toOff + n <= to.length)
    var read = 0
    var endOfStream = false
    while (read < n && !endOfStream) {
      val r = in.read(to, toOff + read, n - read)
      if (r <= 0)
        endOfStream = true
      else
        read += r
    }
    read
  }

  def readRepeatedly(to: Array[Byte]): Int = readRepeatedly(to, 0, to.length)
}
