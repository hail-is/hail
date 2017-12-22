package is.hail.expr.typ

import is.hail.annotations.UnsafeOrdering

/**
  * Created by dking on 12/21/17.
  */
abstract class ComplexType extends Type {
  val representation: Type

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = representation.unsafeOrdering(missingGreatest)

  override def fundamentalType: Type = representation.fundamentalType
}
