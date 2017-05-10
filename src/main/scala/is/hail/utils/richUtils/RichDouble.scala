package is.hail.utils.richUtils

import is.hail.utils.UInt

class RichDouble(val d: Double) extends AnyVal {
  def toUInt: UInt = UInt(d.toLong)
}
