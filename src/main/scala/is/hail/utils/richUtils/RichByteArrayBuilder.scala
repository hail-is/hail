package is.hail.utils.richUtils

import is.hail.utils.ArrayBuilder

import scala.collection.mutable

class RichByteArrayBuilder(val b: ArrayBuilder[Byte]) extends AnyVal {
  def writeULEB128(x0: Int) {
    require(x0 >= 0, s"tried to write negative ULEB value `${ x0 }'")

    var x = x0
    var more = true
    while (more) {
      var c = x & 0x7F
      x = x >>> 7

      if (x == 0)
        more = false
      else
        c = c | 0x80

      assert(c >= 0 && c <= 255)
      b += c.toByte
    }
  }

  def writeSLEB128(x0: Int) {
    var more = true
    var x = x0
    while (more) {
      var c = x & 0x7f
      x >>= 7

      if ((x == 0
        && (c & 0x40) == 0)
        || (x == -1
        && (c & 0x40) == 0x40))
        more = false
      else
        c |= 0x80

      b += c.toByte
    }
  }
}
