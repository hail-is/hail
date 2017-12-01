package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr._

object TypeToTypeInfo {
  def apply(t: Type): TypeInfo[_] = t match {
    case _: TInt32 => typeInfo[Int]
    case _: TInt64 => typeInfo[Long]
    case _: TFloat32 => typeInfo[Float]
    case _: TFloat64 => typeInfo[Double]
    case _: TBoolean => typeInfo[Boolean]
    case _ => typeInfo[Long] // reference types
  }
}
