package is.hail.expr

package object types {
  def coerce[T <: Type](x: Type): T = x.asInstanceOf[T]
}
