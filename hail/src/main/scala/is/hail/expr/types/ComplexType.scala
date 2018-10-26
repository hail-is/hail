package is.hail.expr.types

abstract class ComplexType extends Type {
  val representation: Type

  override def fundamentalType: Type = representation.fundamentalType
}
