package is.hail.expr.types.physical

import is.hail.annotations.UnsafeRow
import is.hail.asm4s.Code

abstract class PIterable extends PType {
  def elementType: PType

  def asPContainer: PContainer = this match {
    case _: PStream => PCanonicalArray(this.elementType, this.required)
    case x: PContainer => x
  }
}
