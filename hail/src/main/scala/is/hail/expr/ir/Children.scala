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
    case UUID4(_) => none
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
    case Consume(value) => FastIndexedSeq(value)
    case If(cond, cnsq, altr) =>
      Array(cond, cnsq, altr)
    case Let(name, value, body) =>
      Array(value, body)
    case RelationalLet(name, value, body) =>
      Array(value, body)
    case AggLet(name, value, body, _) =>
      Array(value, body)
    case TailLoop(_, args, body) =>
      args.map(_._2).toFastIndexedSeq :+ body
    case Recur(_, args, _) =>
      args.toFastIndexedSeq
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
    case ArrayRef(a, i, s) =>
      Array(a, i, s)
    case ArrayLen(a) =>
      Array(a)
    case StreamRange(start, stop, step) =>
      Array(start, stop, step)
    case ArrayZeros(length) =>
      Array(length)
    case MakeNDArray(data, shape, rowMajor) =>
      Array(data, shape, rowMajor)
    case NDArrayShape(nd) =>
      Array(nd)
    case NDArrayReshape(nd, shape) =>
      Array(nd, shape)
    case NDArrayConcat(nds, _) =>
      Array(nds)
    case ArraySort(a, _, _, lessThan) =>
      Array(a, lessThan)
    case ToSet(a) =>
      Array(a)
    case ToDict(a) =>
      Array(a)
    case ToArray(a) =>
      Array(a)
    case CastToArray(a) =>
      Array(a)
    case ToStream(a) =>
      Array(a)
    case LowerBoundOnOrderedCollection(orderedCollection, elem, _) =>
      Array(orderedCollection, elem)
    case GroupByKey(collection) =>
      Array(collection)
    case StreamLen(a) =>
      Array(a)
    case StreamTake(a, len) =>
      Array(a, len)
    case StreamDrop(a, len) =>
      Array(a, len)
    case StreamGrouped(a, size) =>
      Array(a, size)
    case StreamGroupByKey(a, _) =>
      Array(a)
    case StreamMap(a, name, body) =>
      Array(a, body)
    case StreamMerge(l, r, _) =>
      Array(l, r)
    case StreamZip(as, names, body, _) =>
      as :+ body
    case StreamZipJoin(as, _, _, _, joinF) =>
      as :+ joinF
    case StreamMultiMerge(as, _) =>
      as
    case StreamFilter(a, name, cond) =>
      Array(a, cond)
    case StreamFlatMap(a, name, body) =>
      Array(a, body)
    case StreamFold(a, zero, accumName, valueName, body) =>
      Array(a, zero, body)
    case StreamFold2(a, accum, valueName, seq, result) =>
      Array(a) ++ accum.map(_._2) ++ seq ++ Array(result)
    case StreamScan(a, zero, accumName, valueName, body) =>
      Array(a, zero, body)
    case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
      Array(left, right, join)
    case StreamFor(a, valueName, body) =>
      Array(a, body)
    case StreamAgg(a, name, query) =>
      Array(a, query)
    case StreamAggScan(a, name, query) =>
      Array(a, query)
    case RunAggScan(array, _, init, seq, result, _) =>
      Array(array, init, seq, result)
    case RunAgg(body, result, _) =>
      Array(body, result)
    case NDArrayRef(nd, idxs) =>
      nd +: idxs
    case NDArraySlice(nd, slices) =>
      Array(nd, slices)
    case NDArrayFilter(nd, keep) =>
      nd +: keep
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
    case NDArrayQR(nd, _) =>
      Array(nd)
    case NDArrayInv(nd) =>
      Array(nd)
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
    case InitOp(_, args, _) => args
    case SeqOp(_, args, _) => args
    case _: ResultOp => none
    case _: AggStateValue => none
    case _: CombOp => none
    case CombOpValue(_, value, _) => Array(value)
    case InitFromSerializedValue(_, value, _) => Array(value)
    case SerializeAggs(_, _, _, _) => none
    case DeserializeAggs(_, _, _, _) => none
    case Begin(xs) =>
      xs
    case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
      initOpArgs ++ seqOpArgs
    case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
      initOpArgs ++ seqOpArgs
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
    case ApplyIR(_, _, args) =>
      args.toFastIndexedSeq
    case Apply(_, _, args, _) =>
      args.toFastIndexedSeq
    case ApplySeeded(_, args, seed, _) =>
      args.toFastIndexedSeq
    case ApplySpecial(_, _, args, _) =>
      args.toFastIndexedSeq
    // from MatrixIR
    case MatrixWrite(child, _) => Array(child)
    case MatrixMultiWrite(children, _) => children
    // from TableIR
    case TableCount(child) => Array(child)
    case MatrixCount(child) => Array(child)
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
    case BlockMatrixCollect(child) => Array(child)
    case BlockMatrixWrite(child, _) => Array(child)
    case UnpersistBlockMatrix(child) => Array(child)
    case BlockMatrixMultiWrite(blockMatrices, _) => blockMatrices
    case CollectDistributedArray(ctxs, globals, _, _, body) => Array(ctxs, globals, body)
    case ReadPartition(path, _, _) => Array(path)
    case WritePartition(stream, ctx, _) => Array(stream, ctx)
    case WriteMetadata(writeAnnotations, _) => Array(writeAnnotations)
    case ReadValue(path, _, _) => Array(path)
    case WriteValue(value, pathPrefix, spec) => Array(value, pathPrefix)
    case LiftMeOut(child) => Array(child)
    case ShuffleWith(keyFields, rowType, rowEType, keyEType, name, writer, readers) =>
      Array(writer, readers)
    case ShuffleWrite(id, rows) =>
      Array(id, rows)
    case ShufflePartitionBounds(id, nPartitions) =>
      Array(id, nPartitions)
    case ShuffleRead(id, keyRange) =>
      Array(id, keyRange)
  }
}
