package is.hail.expr.ir

import is.hail.utils._

object Children {
  private val none: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def apply(x: IR): IndexedSeq[BaseIR] = x match {
    case I32(x) => none
    case I64(x) => none
    case F32(x) => none
    case F64(x) => none
    case Str(x) => none
    case True() => none
    case False() => none
    case Literal(_, _) => none
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
    case AggLet(name, value, body) =>
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
      args.toFastIndexedSeq
    case ArrayRef(a, i) =>
      Array(a, i)
    case ArrayLen(a) =>
      Array(a)
    case ArrayRange(start, stop, step) =>
      Array(start, stop, step)
    case MakeNDArray(data, shape, row_major) =>
      Array(data, shape, row_major)
    case ArraySort(a, _, _, compare) =>
      Array(a, compare)
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
    case ArrayScan(a, zero, accumName, valueName, body) =>
      Array(a, zero, body)
    case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
      Array(left, right, compare, join)
    case ArrayFor(a, valueName, body) =>
      Array(a, body)
    case ArrayAgg(a, name, query) =>
      Array(a, query)
    case NDArrayRef(nd, idxs) =>
      Array(nd, idxs)
    case AggFilter(cond, aggIR) =>
      Array(cond, aggIR)
    case AggExplode(array, _, aggBody) =>
      Array(array, aggBody)
    case AggGroupBy(key, aggIR) =>
      Array(key, aggIR)
    case AggArrayPerElement(a, name, aggBody) => Array(a, aggBody)
    case MakeStruct(fields) =>
      fields.map(_._2).toFastIndexedSeq
    case SelectFields(old, fields) =>
      Array(old)
    case InsertFields(old, fields, _) =>
      (old +: fields.map(_._2)).toFastIndexedSeq
    case InitOp(i, args, aggSig) =>
      i +: args
    case SeqOp(i, args, _) =>
      i +: args
    case Begin(xs) =>
      xs
    case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
      constructorArgs ++ initOpArgs.getOrElse(FastIndexedSeq()) ++ seqOpArgs
    case ApplyScanOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
      constructorArgs ++ initOpArgs.getOrElse(FastIndexedSeq()) ++ seqOpArgs
    case GetField(o, name) =>
      Array(o)
    case MakeTuple(types) =>
      types.toFastIndexedSeq
    case GetTupleElement(o, idx) =>
      Array(o)
    case StringSlice(s, start, n) =>
      Array(s, start, n)
    case StringLength(s) =>
      Array(s)
    case In(i, typ) =>
      none
    case Die(message, typ) =>
      Array(message)
    case ApplyIR(_, args) =>
      args.toFastIndexedSeq
    case Apply(_, args) =>
      args.toFastIndexedSeq
    case ApplySeeded(_, args, seed) =>
      args.toFastIndexedSeq
    case ApplySpecial(_, args) =>
      args.toFastIndexedSeq
    case Uniroot(_, fn, min, max) =>
      FastIndexedSeq(fn, min, max)
    // from MatrixIR
    case MatrixWrite(child, _) => IndexedSeq(child)
    case MatrixMultiWrite(children, _) => children
    // from TableIR
    case TableCount(child) => IndexedSeq(child)
    case TableGetGlobals(child) => IndexedSeq(child)
    case TableCollect(child) => IndexedSeq(child)
    case TableAggregate(child, query) => IndexedSeq(child, query)
    case MatrixAggregate(child, query) => IndexedSeq(child, query)
    case TableWrite(child, _, _, _, _) => IndexedSeq(child)
    case TableExport(child, _, _, _, _, _) => IndexedSeq(child)
    case TableToValueApply(child, _) => IndexedSeq(child)
    case MatrixToValueApply(child, _) => IndexedSeq(child)
    // from BlockMatrixIR
    case BlockMatrixToValueApply(child, _) => IndexedSeq(child)
    case BlockMatrixWrite(child, _) => IndexedSeq(child)
    case CollectDistributedArray(ctxs, globals, _, _, body) => IndexedSeq(ctxs, globals, body)
  }
}