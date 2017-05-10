package is.hail.utils.richUtils

import is.hail.utils.UInt

class RichInt(val i: Int) extends AnyVal {
  def toUInt: UInt = UInt(i)

  def toUIntNoCheck: UInt = UInt.applyUnchecked(i)
}
