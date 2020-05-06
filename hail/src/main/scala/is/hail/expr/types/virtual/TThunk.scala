package is.hail.expr.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.physical.CType

import scala.reflect.ClassTag

case class TThunk(ct: CType) extends Type {
  def ordering: ExtendedOrdering = ???

  def _typeCheck(a: Any): Boolean = ???

  def _toPretty = s"Thunk[$ct]"

  def scalaClassTag: ClassTag[_ <: AnyRef] = ???
}
