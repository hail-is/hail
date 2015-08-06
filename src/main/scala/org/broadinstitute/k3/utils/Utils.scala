package org.broadinstitute.k3.utils

import scala.collection.mutable.ArrayBuilder

object Utils {
  def writeULEB128(b: ArrayBuilder[Byte], x0: Int) {
    require(x0 >= 0)

    // FIXME functionalize
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

  def fail(): Unit = {
    assert(false)
  }
}
