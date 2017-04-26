package is.hail.utils.richUtils

import is.hail.utils.UInt

class RichLong(val l: Long) extends AnyVal {
  def toUInt: UInt = {
    assert((l >>> 32) == 0, s"High bits not set to 0. Cannot convert ${ l.toBinaryString } to UInt.")
    UInt(l.toInt)
  }
}
