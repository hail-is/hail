package is.hail.expr.types.physical

import is.hail.annotations.UnsafeOrdering

abstract class ComplexPType extends PType {
  val representation: PType

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override def fundamentalType: PType = representation.fundamentalType

  override def containsPointers: Boolean = representation.containsPointers
}
