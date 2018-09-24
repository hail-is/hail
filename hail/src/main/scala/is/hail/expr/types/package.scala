package is.hail.expr

import is.hail.expr.types.physical.PType

package object types {
  def coerce[T <: Type](x: Type): T = x.asInstanceOf[T]

  def coerce[T <: PType](x: PType): T = x.asInstanceOf[T]
}
