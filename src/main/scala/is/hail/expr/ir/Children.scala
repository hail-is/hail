package is.hail.expr.ir

import is.hail.expr.BaseIR

object Children {
  private def none: IndexedSeq[BaseIR] = Array.empty[BaseIR]
  def apply(x: IR): IndexedSeq[BaseIR] = x match {
    case I32(x) => none
    case I64(x) => none
    case F32(x) => none
    case F64(x) => none
    case True() => none
    case False() => none
    case NA(typ) => none
    case MapNA(name, value, body, typ) =>
      Array(value, body)
    case IsNA(value) =>
      Array(value)
    case If(cond, cnsq, altr, typ) =>
      Array(cond, cnsq, altr)
    case Let(name, value, body, typ) =>
      Array(value, body)
    case Ref(name, typ) =>
      none
    case ApplyPrimitive(op, args, typ) =>
      args
    case LazyApplyPrimitive(op, args, typ) =>
      args
    case Lambda(names, body, typ) =>
      Array(body)
    case MakeArray(args, typ) =>
      args
    case MakeArrayN(len, elementType) =>
      Array(len)
    case ArrayRef(a, i, typ) =>
      Array(a, i)
    case ArrayMissingnessRef(a, i) =>
      Array(a, i)
    case ArrayLen(a) =>
      Array(a)
    case ArrayMap(a, lam, elementTyp) =>
      Array(a, lam)
    case ArrayFold(a, zero, lam, typ) =>
      Array(a, zero, lam)
    case MakeStruct(fields, missingness) =>
      fields.map(_._3)
    case GetField(o, name, typ) =>
      Array(o)
    case GetFieldMissingness(o, name) =>
      Array(o)
    case In(i, typ) =>
      none
    case InMissingness(i) =>
      none
    case Out(v) =>
      Array(v)
    case Die(message) =>
      none
  }
}
