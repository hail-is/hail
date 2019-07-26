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
    case CastRename(v, typ) =>
      Array(v)
    case NA(typ) => none
    case IsNA(value) =>
      Array(value)
    case Coalesce(values) => values.toFastIndexedSeq
    case If(cond, cnsq, altr) =>
      Array(cond, cnsq, altr)
    case Let(name, value, body) =>
      Array(value, body)
    case RelationalLet(name, value, body) =>
      Array(value, body)
    case AggLet(name, value, body, _) =>
      Array(value, body)
    case Ref(name, typ) =>
      none
    case RelationalRef(_, _) =>
      none
    case ApplyBinaryPrimOp(op, l, r) =>
      Array(l, r)
    case ApplyUnaryPrimOp(op, x) =>
      Array(x)
    case ApplyComparisonOp(op, l, r) =>
      Array(l, r)
    case MakeArray(args, typ) =>
      args.toFastIndexedSeq
    case MakeStream(args, typ) =>
      args.toFastIndexedSeq
    case ArrayRef(a, i) =>
      Array(a, i)
    case ArrayLen(a) =>
      Array(a)
    case ArrayRange(start, stop, step) =>
      Array(start, stop, step)
    case StreamRange(start, stop, step) =>
      Array(start, stop, step)
    case MakeNDArray(data, shape, rowMajor) =>
      Array(data, shape, rowMajor)
    case NDArrayShape(nd) =>
      Array(nd)
    case NDArrayReshape(nd, shape) =>
      Array(nd, shape)
    case ArraySort(a, _, _, compare) =>
      Array(a, compare)
    case ToSet(a) =>
      Array(a)
    case ToDict(a) =>
      Array(a)
    case ToArray(a) =>
      Array(a)
    case ToStream(a) =>
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
    case ArrayAggScan(a, name, query) =>
      Array(a, query)
    case NDArrayRef(nd, idxs) =>
      nd +: idxs
    case NDArraySlice(nd, slices) =>
      Array(nd, slices)
    case NDArrayMap(nd, _, body) =>
      Array(nd, body)
    case NDArrayMap2(l, r, _, _, body) =>
      Array(l, r, body)
    case NDArrayReindex(nd, _) =>
      Array(nd)
    case NDArrayAgg(nd, _) =>
      Array(nd)
    case NDArrayMatMul(l, r) =>
      Array(l, r)
    case NDArrayWrite(nd, path) =>
      Array(nd, path)
    case AggFilter(cond, aggIR, _) =>
      Array(cond, aggIR)
    case AggExplode(array, _, aggBody, _) =>
      Array(array, aggBody)
    case AggGroupBy(key, aggIR, _) =>
      Array(key, aggIR)
    case AggArrayPerElement(a, _, _, aggBody, knownLength, _) => Array(a, aggBody) ++ knownLength.toArray[IR]
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
    case InitOp2(_, args, _) => args
    case SeqOp2(_, args, _) => args
    case _: ResultOp2 => none
    case _: CombOp2 => none
    case SerializeAggs(_, _, _, _) => none
    case DeserializeAggs(_, _, _, _) => none
    case Begin(xs) =>
      xs
    case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
      constructorArgs ++ initOpArgs.getOrElse(FastIndexedSeq()) ++ seqOpArgs
    case ApplyScanOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
      constructorArgs ++ initOpArgs.getOrElse(FastIndexedSeq()) ++ seqOpArgs
    case GetField(o, name) =>
      Array(o)
    case MakeTuple(fields) =>
      fields.map(_._2).toFastIndexedSeq
    case GetTupleElement(o, idx) =>
      Array(o)
    case In(i, typ) =>
      none
    case Die(message, typ) =>
      Array(message)
    case ApplyIR(_, args) =>
      args.toFastIndexedSeq
    case Apply(_, args, _) =>
      args.toFastIndexedSeq
    case ApplySeeded(_, args, seed, _) =>
      args.toFastIndexedSeq
    case ApplySpecial(_, args, _) =>
      args.toFastIndexedSeq
    case Uniroot(_, fn, min, max) =>
      Array(fn, min, max)
    // from MatrixIR
    case MatrixWrite(child, _) => Array(child)
    case MatrixMultiWrite(children, _) => children
    // from TableIR
    case TableCount(child) => Array(child)
    case TableGetGlobals(child) => Array(child)
    case TableCollect(child) => Array(child)
    case TableAggregate(child, query) => Array(child, query)
    case MatrixAggregate(child, query) => Array(child, query)
    case TableWrite(child, _) => Array(child)
    case TableMultiWrite(children, _) => children
    case TableToValueApply(child, _) => Array(child)
    case MatrixToValueApply(child, _) => Array(child)
    // from BlockMatrixIR
    case BlockMatrixToValueApply(child, _) => Array(child)
    case BlockMatrixWrite(child, _) => Array(child)
    case BlockMatrixMultiWrite(blockMatrices, _) => blockMatrices
    case CollectDistributedArray(ctxs, globals, _, _, body) => Array(ctxs, globals, body)
    case ReadPartition(path, _, _) => Array(path)
  }
}