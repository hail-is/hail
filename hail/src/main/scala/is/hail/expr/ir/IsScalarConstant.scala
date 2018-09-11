package is.hail.expr.ir

import is.hail.expr.types._

object IsScalarType {
  def apply(t: Type): Boolean = {
    t match {
      case TInt32(_) | TInt64(_) | TFloat32(_) | TFloat64(_) | TBoolean(_) => true
      case _ => false
    }
  }
}


object IsScalarConstant {
  def apply(ir: IR): Boolean = {
    ir match {
      case I32(_) | I64(_) | F32(_) | F64(_) | True() | False() => true
      case NA(t) if IsScalarType(t) => true
      case _ => false
    }
  }
}
