package is.hail.expr.types

abstract class PhysicalType {
  def virtualType: Type
}

case class PDefault(virtualType: Type) extends PhysicalType
