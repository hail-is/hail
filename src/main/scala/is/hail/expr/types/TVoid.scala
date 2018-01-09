package is.hail.expr.types

case object TVoid extends Type {
  override val required = true

  override def _toString = "Void"

  override def ordering(missingGreatest: Boolean): Ordering[is.hail.annotations.Annotation] = throw new UnsupportedOperationException("No ordering on Void")

  override def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] = throw new UnsupportedOperationException("No ClassTag for Void")

  override def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException("No elements of Void")

  override def isRealizable = false
}
