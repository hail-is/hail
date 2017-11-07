package is.hail.expr

import is.hail.asm4s._

package object ir {
  def coerce[T](x: Code[_]): Code[T] = x.asInstanceOf[Code[T]]
}
