package is.hail.expr.ir

import is.hail.asm4s.Settable
import is.hail.types.virtual.{Type, TInt32, TInt64, TFloat32, TFloat64}

object NumericPrimitives {

  def newLocal(cb: EmitCodeBuilder, name: String, typ: Type): Settable[Any] = {
    coerce[Any](typ match {
      case TInt32 => cb.newLocal[Int](name)
      case TInt64 => cb.newLocal[Long](name)
      case TFloat32 => cb.newLocal[Float](name)
      case TFloat64 => cb.newLocal[Double](name)
    })
  }

}
