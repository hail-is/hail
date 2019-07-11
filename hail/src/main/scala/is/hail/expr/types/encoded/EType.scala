package is.hail.expr.types.encoded

import is.hail.expr.types.virtual.Type

abstract class EType {
  def required: Boolean
}
