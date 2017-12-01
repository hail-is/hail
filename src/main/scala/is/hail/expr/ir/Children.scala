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
    case Cast(v, typ) =>
      Array(v)
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
    case ApplyBinaryPrimOp(op, l, r, typ) =>
      Array(l, r)
    case ApplyUnaryPrimOp(op, x, typ) =>
      Array(x)
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
    case ArrayMap(a, name, body, elementTyp) =>
      Array(a, body)
    case ArrayFold(a, zero, accumName, valueName, body, typ) =>
      Array(a, zero, body)
    case MakeStruct(fields) =>
      fields.map(_._3)
    case AggIn(_) =>
      none
    case AggMap(a, _, body, _) =>
      Array(a, body)
    case AggSum(a, _) =>
      Array(a)
    case GetField(o, name, typ) =>
      Array(o)
    case GetFieldMissingness(o, name) =>
      Array(o)
    case In(i, typ) =>
      none
    case InMissingness(i) =>
      none
    case Die(message) =>
      none
  }
}
