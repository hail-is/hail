package is.hail.expr.ir

import is.hail.utils.{FastIndexedSeq, FastSeq}

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
    case ApplyComparisonOp(op, l, r) =>
      Array(l, r)
    case MakeArray(args, typ) =>
      args.toIndexedSeq
    case ArrayRef(a, i) =>
      Array(a, i)
    case ArrayLen(a) =>
      Array(a)
    case ArrayRange(start, stop, step) =>
      Array(start, stop, step)
    case ArraySort(a, ascending, _) =>
      Array(a, ascending)
    case ToSet(a) =>
      Array(a)
    case ToDict(a) =>
      Array(a)
    case ToArray(a) =>
      Array(a)
    case LowerBoundOnOrderedCollection(orderedCollection, elem, _) =>
      Array(orderedCollection, elem)
    case GroupByKey(collection) =>
      Array(collection)
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
    case SelectFields(old, fields) =>
      Array(old)
    case InsertFields(old, fields) =>
      (old +: fields.map(_._2)).toIndexedSeq
    case InitOp(i, args, aggSig) =>
      i +: args
    case SeqOp(i, args, _) =>
      i +: args
    case Begin(xs) =>
      xs
    case ApplyAggOp(a, constructorArgs, initOpArgs, aggSig) =>
      (a +: constructorArgs) ++ initOpArgs.getOrElse(FastIndexedSeq())
    case ApplyScanOp(a, constructorArgs, initOpArgs, aggSig) =>
      (a +: constructorArgs) ++ initOpArgs.getOrElse(FastIndexedSeq())
    case GetField(o, name) =>
      Array(o)
    case MakeTuple(types) =>
      types.toIndexedSeq
    case GetTupleElement(o, idx) =>
      Array(o)
    case StringSlice(s, start, n) =>
      Array(s, start, n)
    case StringLength(s) =>
      Array(s)
    case In(i, typ) =>
      none
    case Die(message, typ) =>
      none
    case ApplyIR(_, args, _) =>
      args.toIndexedSeq
    case Apply(_, args) =>
      args.toIndexedSeq
    case ApplySpecial(_, args) =>
      args.toIndexedSeq
    case Uniroot(_, fn, min, max) =>
      FastIndexedSeq(fn, min, max)
    // from MatrixIR
    case MatrixWrite(child, _) => IndexedSeq(child)
    // from TableIR
    case TableCount(child) => IndexedSeq(child)
    case TableAggregate(child, query) => IndexedSeq(child, query)
    case TableWrite(child, _, _, _) => IndexedSeq(child)
    case TableExport(child, _, _, _, _) => IndexedSeq(child)
  }
}
