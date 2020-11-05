package is.hail.utils.richUtils

import java.io.InputStream
import is.hail.utils._

class UnexpectedEndOfFileHailException(
  msg: String,
  logMsg: Option[String] = None,
  cause: Throwable = null,
  errorId: Int = -1
) extends HailException(msg, logMsg, cause, errorId) {
}

class RichInputStream(val in: InputStream) extends AnyVal {
  def readFully(to: Array[Byte]): Unit = {
    readFully(to, 0, to.length)
  }

  def readFully(to: Array[Byte], toOff: Int, n: Int): Unit = {
    val nRead = readRepeatedly(to, toOff, n)
    if (nRead < n) throw new UnexpectedEndOfFileHailException(
      s"Premature end of file: expected $n bytes, found $nRead")
  }

  def readRepeatedly(to: Array[Byte], toOff: Int, n: Int): Int = {
    assert(toOff + n <= to.length)
    var read = 0
    var endOfStream = false
    while (read < n && !endOfStream) {
      val r = in.read(to, toOff + read, n - read)
      if (r < 0)
        endOfStream = true
      else
        read += r
    }
    read
  }

  def readRepeatedly(to: Array[Byte]): Int = readRepeatedly(to, 0, to.length)
}
