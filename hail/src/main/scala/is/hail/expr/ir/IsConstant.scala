package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual._

object CanEmit {
  def apply(t: Type): Boolean = {
    t match {
      case TInt32(_) | TInt64(_) | TFloat32(_) | TFloat64(_) | TBoolean(_) | TString(_) => true
      case _ => false
    }
  }
}

object IsConstant {
  def apply(ir: IR): Boolean = {
    ir match {
      case I32(_) | I64(_) | F32(_) | F64(_) | True() | False() | NA(_) | Str(_) | Literal(_, _) => true
      case _ => false
    }
  }
}
