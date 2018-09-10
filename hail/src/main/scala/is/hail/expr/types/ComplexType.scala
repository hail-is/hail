package is.hail.expr.types

abstract class ComplexType extends Type {
  val representation: Type

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def fundamentalType: Type = representation.fundamentalType
}
