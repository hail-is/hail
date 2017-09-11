package is.hail.expr

import is.hail.expr.ir.PointedTypeInfo._

package object ir {
  def Out1(x: IR) = new Out(Array(x))
  def _NA[T: PointedTypeInfo] = new NA(pointedTypeInfo[T])
}
