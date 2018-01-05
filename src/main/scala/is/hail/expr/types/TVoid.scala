package is.hail.expr.types

import is.hail.annotations.ExtendedOrdering

case object TVoid extends Type {
  override val required = true

  override def _toString = "Void"

  val ordering: ExtendedOrdering = null

  override def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] = throw new UnsupportedOperationException("No ClassTag for Void")

  override def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException("No elements of Void")

  override def isRealizable = false
}
