package is.hail.utils.richUtils

import is.hail.utils.UInt

class RichLong(val l: Long) extends AnyVal {
  def toUInt: UInt = UInt(l)
}
