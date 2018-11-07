package is.hail.expr.types

import is.hail.expr.types.virtual.Type

abstract class EncodedType {
  def virtualType: Type
}

case class EDefault(virtualType: Type) extends EncodedType
