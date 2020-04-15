package is.hail.shuffler

import java.util.Comparator

class SameLengthByteArrayComparator extends Comparator[Array[Byte]] {
  def compare(l: Array[Byte], r: Array[Byte]): Int = {
    assert(l.length == r.length)
    val length = l.length
    var i = 0
    while (i < length) {
      if (l(i) < r(i)) {
        return -1
      }
      if (l(i) > r(i)) {
        return 1
      }
      i += 1
    }
    return 0
  }
}
