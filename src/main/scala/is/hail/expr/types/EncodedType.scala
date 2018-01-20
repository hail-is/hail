package is.hail.expr.types

abstract class EncodedType {
  def virtualType: Type
}

case class EDefault(virtualType: Type) extends EncodedType
