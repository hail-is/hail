package is.hail.expr.ir

import is.hail.expr.BaseIR

object Children {
  private def none: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def apply(x: IR): IndexedSeq[BaseIR] = x match {
    case I32(x) => none
    case I64(x) => none
    case F32(x) => none
    case F64(x) => none
    case Str(x) => none
    case True() => none
    case False() => none
    case Void() => none
    case Cast(v, typ) =>
      Array(v)
    case NA(typ) => none
    case IsNA(value) =>
      Array(value)
    case If(cond, cnsq, altr) =>
      Array(cond, cnsq, altr)
    case Let(name, value, body) =>
      Array(value, body)
    case Ref(name, typ) =>
      none
    case ApplyBinaryPrimOp(op, l, r) =>
      Array(l, r)
    case ApplyUnaryPrimOp(op, x) =>
      Array(x)
    case MakeArray(args, typ) =>
      args.toIndexedSeq
    case ArrayRef(a, i) =>
      Array(a, i)
    case ArrayLen(a) =>
      Array(a)
    case ArrayRange(start, stop, step) =>
      Array(start, stop, step)
    case ArrayMap(a, name, body) =>
      Array(a, body)
    case ArrayFilter(a, name, cond) =>
      Array(a, cond)
    case ArrayFlatMap(a, name, body) =>
      Array(a, body)
    case ArrayFold(a, zero, accumName, valueName, body) =>
      Array(a, zero, body)
    case ArrayFor(a, valueName, body) =>
      Array(a, body)
    case MakeStruct(fields) =>
      fields.map(_._2).toIndexedSeq
    case InsertFields(old, fields) =>
      (old +: fields.map(_._2)).toIndexedSeq
    case AggIn(_) =>
      none
    case AggMap(a, _, body) =>
      Array(a, body)
    case AggFilter(a, name, body) =>
      Array(a, body)
    case AggFlatMap(a, name, body) =>
      Array(a, body)
    case SeqOp(a, i, _) =>
      Array(a, i)
    case Begin(xs) =>
      xs
    case ApplyAggOp(a, op, args) =>
      (a +: args).toIndexedSeq
    case GetField(o, name) =>
      Array(o)
    case MakeTuple(types) =>
      types.toIndexedSeq
    case GetTupleElement(o, idx) =>
      Array(o)
    case In(i, typ) =>
      none
    case Die(message) =>
      none
    case Apply(_, args) =>
      args.toIndexedSeq
    case ApplySpecial(_, args) =>
      args.toIndexedSeq
    // from MatrixIR
    case MatrixWrite(child, _, _, _) => IndexedSeq(child)
    case ExportPlink(child, _) => IndexedSeq(child)
    // from TableIR
    case TableCount(child) => IndexedSeq(child)
    case TableAggregate(child, query) => IndexedSeq(child, query)
    case TableWrite(child, _, _, _) => IndexedSeq(child)
    case TableExport(child, _, _, _, _) => IndexedSeq(child)
  }
}
