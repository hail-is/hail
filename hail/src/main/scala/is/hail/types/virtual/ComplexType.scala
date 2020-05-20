package is.hail.types.virtual

abstract class ComplexType extends Type {
  val representation: Type

  override def fundamentalType: Type = representation.fundamentalType
}
