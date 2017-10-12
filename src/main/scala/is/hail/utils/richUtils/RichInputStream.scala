package is.hail.utils.richUtils

import java.io.InputStream

class RichInputStream(val in: InputStream) extends AnyVal {
  def readFully(to: Array[Byte], toOff: Int, n: Int) {
    assert(toOff + n <= to.length)

    var read = 0
    while (read < n) {
      val r = in.read(to, toOff + read, n - read)
      assert(r > 0)
      read += r
    }
  }
}
