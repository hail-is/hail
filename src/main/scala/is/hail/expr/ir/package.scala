package is.hail.expr

import is.hail.asm4s._

package object ir {
  def coerce[T](x: Code[_]): Code[T] = x.asInstanceOf[Code[T]]

  def defaultValue(t: Type): Code[_] = t match {
    case _: TBoolean => false
    case _: TInt32 => 0
    case _: TInt64 => 0L
    case _: TFloat32 => 0.0f
    case _: TFloat64 => 0.0
    case _ => 0L // reference types
  }
}
