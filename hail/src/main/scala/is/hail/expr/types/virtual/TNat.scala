package is.hail.expr.types.virtual
import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.physical.PType

import scala.reflect.ClassTag

final case class TNat(n: Int, override val required: Boolean = false) extends Type {
  override def physicalType: PType = throw new UnsupportedOperationException

  override def _toPretty: String = n.toString

  override def scalaClassTag: ClassTag[_ <: AnyRef] = throw new UnsupportedOperationException

  override val ordering: ExtendedOrdering = null

  override def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException
}
