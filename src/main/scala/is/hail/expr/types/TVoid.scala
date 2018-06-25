package is.hail.expr.types

import is.hail.annotations.{CodeOrdering, ExtendedOrdering}
import is.hail.expr.ir.EmitMethodBuilder

case object TVoid extends Type {
  override val required = true

  override def _toPretty = "Void"

  val ordering: ExtendedOrdering = null

  override def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] = throw new UnsupportedOperationException("No ClassTag for Void")

  override def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException("No elements of Void")

  override def isRealizable = false

  def codeOrdering(mb: EmitMethodBuilder, other: Type): CodeOrdering = null
}
