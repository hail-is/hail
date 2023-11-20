package is.hail.expr.ir

import is.hail.types.virtual._

object CanEmit {
  def apply(t: Type): Boolean = {
    t match {
      case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean | TString => true
      case _ => false
    }
  }
}

object Constant {
  def unapply(ir: IR): Option[IR] =
    Some(ir).filter(IsConstant(_))
}

object IsConstant {
  def apply(ir: IR): Boolean = {
    ir match {
      case I32(_) | I64(_) | F32(_) | F64(_) | True() | False() | NA(_) | Str(_) | Literal(_, _) | EncodedLiteral(_, _) => true
      case _ => false
    }
  }
}
