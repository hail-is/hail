package is.hail.expr.types.virtual
import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.physical.{PNat, PType}

import scala.reflect.ClassTag

final case class TNat(n: Int, override val required: Boolean = false) extends Type {
  override def physicalType: PType = PNat(n)

  override def _toPretty: String = s"nat<$n>"

  override def scalaClassTag: ClassTag[_ <: AnyRef] = throw new UnsupportedOperationException

  override val ordering: ExtendedOrdering = null

  override def _typeCheck(a: Any): Boolean = (a.isInstanceOf[Int] || a.isInstanceOf[Long]) && a == n
}
