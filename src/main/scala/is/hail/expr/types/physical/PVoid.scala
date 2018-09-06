package is.hail.expr.types.physical
import is.hail.annotations.{CodeOrdering, ExtendedOrdering}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.{TVoid, Type}

case object PVoid extends PType {
  def virtualType: Type = TVoid

  override val required = true

  override def _toPretty = "Void"

  val ordering: ExtendedOrdering = null

  override def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] = throw new UnsupportedOperationException("No ClassTag for Void")

  override def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException("No elements of Void")

  override def isRealizable = false

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = null
}
