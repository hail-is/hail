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
    case EncodedLiteral(_, _) => none
    case Void() => none
    case Cast(v, typ) =>
      Array(v)
    case CastRename(v, typ) =>
      Array(v)
    case NA(typ) => none
    case IsNA(value) =>
      Array(value)
    case Coalesce(values) => values.toFastSeq
    case Consume(value) => FastSeq(value)
    case If(cond, cnsq, altr) =>
      Array(cond, cnsq, altr)
    case s@Switch(x, default, cases) =>
      val children = Array.ofDim[BaseIR](s.size)
      children(0) = x
      children(1) = default
      for (i <- cases.indices) children(2 + i) = cases(i)
      children
    case Let(bindings, body) =>
      val children = Array.ofDim[BaseIR](x.size)
      for (i <- bindings.indices) children(i) = bindings(i)._2
      children(bindings.size) = body
      children
    case RelationalLet(name, value, body) =>
      Array(value, body)
    case AggLet(name, value, body, _) =>
      Array(value, body)
    case TailLoop(_, args, body) =>
      args.map(_._2).toFastSeq :+ body
    case Recur(_, args, _) =>
      args.toFastSeq
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
      args.toFastSeq
    case MakeStream(args, _, _) =>
      args.toFastSeq
    case ArrayRef(a, i, _) =>
      Array(a, i)
    case ArraySlice(a, start, stop, step, _) =>
      if (stop.isEmpty)
        Array(a, start, step)
      else
        Array(a, start, stop.get, step)
    case ArrayLen(a) =>
      Array(a)
    case StreamIota(start, step, _) =>
      Array(start, step)
    case StreamRange(start, stop, step, _, _) =>
      Array(start, stop, step)
    case SeqSample(totalRange, numToSample, rngState, _) =>
      Array(totalRange, numToSample, rngState)
    case StreamDistribute(child, pivots, path, _, _) =>
      Array(child, pivots, path)
    case StreamWhiten(stream, _, _, _, _, _, _, _) =>
      Array(stream)
    case ArrayZeros(length) =>
      Array(length)
    case MakeNDArray(data, shape, rowMajor, _) =>
      Array(data, shape, rowMajor)
    case NDArrayShape(nd) =>
      Array(nd)
    case NDArrayReshape(nd, shape, _) =>
      Array(nd, shape)
    case NDArrayConcat(nds, _) =>
      Array(nds)
    case ArraySort(a, _, _, lessThan) =>
      Array(a, lessThan)
    case ArrayMaximalIndependentSet(a, tieBreaker) =>
      Array(a) ++ tieBreaker.map { case (_, _, tb) => tb }
    case ToSet(a) =>
      Array(a)
    case ToDict(a) =>
      Array(a)
    case ToArray(a) =>
      Array(a)
    case CastToArray(a) =>
      Array(a)
    case ToStream(a, _) =>
      Array(a)
    case LowerBoundOnOrderedCollection(orderedCollection, elem, _) =>
      Array(orderedCollection, elem)
    case GroupByKey(collection) =>
      Array(collection)
    case RNGStateLiteral() => none
    case RNGSplit(state, split) =>
      Array(state, split)
    case StreamLen(a) =>
      Array(a)
    case StreamTake(a, len) =>
      Array(a, len)
    case StreamDrop(a, len) =>
      Array(a, len)
    case StreamGrouped(a, size) =>
      Array(a, size)
    case StreamGroupByKey(a, _, _) =>
      Array(a)
    case StreamMap(a, name, body) =>
      Array(a, body)
    case StreamZip(as, names, body, _, _) =>
      as :+ body
    case StreamZipJoin(as, _, _, _, joinF) =>
      as :+ joinF
    case StreamZipJoinProducers(contexts, _, makeProducer, _, _, _, joinF) =>
      Array(contexts, makeProducer, joinF)
    case StreamMultiMerge(as, _) =>
      as
    case StreamFilter(a, name, cond) =>
      Array(a, cond)
    case StreamTakeWhile(a, name, cond) =>
      Array(a, cond)
    case StreamDropWhile(a, name, cond) =>
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
    case StreamBufferedAggregate(streamChild, initAggs, newKey, seqOps, _, _, _) =>
      Array(streamChild, initAggs, newKey, seqOps)
    case StreamLocalLDPrune(streamChild, r2Threshold, windowSize, maxQueueSize, nSamples) =>
      Array(streamChild, r2Threshold, windowSize, maxQueueSize, nSamples)
    case RunAggScan(array, _, init, seq, result, _) =>
      Array(array, init, seq, result)
    case RunAgg(body, result, _) =>
      Array(body, result)
    case NDArrayRef(nd, idxs, _) =>
      nd +: idxs
    case NDArraySlice(nd, slices) =>
      Array(nd, slices)
    case NDArrayFilter(nd, keep) =>
      nd +: keep
    case NDArrayMap(nd, _, body) =>
      Array(nd, body)
    case NDArrayMap2(l, r, _, _, body, _) =>
      Array(l, r, body)
    case NDArrayReindex(nd, _) =>
      Array(nd)
    case NDArrayAgg(nd, _) =>
      Array(nd)
    case NDArrayMatMul(l, r, _) =>
      Array(l, r)
    case NDArrayQR(nd, _, _) =>
      Array(nd)
    case NDArraySVD(nd, _, _, _) =>
      Array(nd)
    case NDArrayEigh(nd, _, _) =>
      Array(nd)
    case NDArrayInv(nd, errorID) =>
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
      fields.map(_._2).toFastSeq
    case SelectFields(old, fields) =>
      Array(old)
    case InsertFields(old, fields, _) =>
      (old +: fields.map(_._2)).toFastSeq
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
    case AggFold(zero, seqOp, combOp, elementName, accumName, _) =>
      Array(zero, seqOp, combOp)
    case GetField(o, name) =>
      Array(o)
    case MakeTuple(fields) =>
      fields.map(_._2).toFastSeq
    case GetTupleElement(o, idx) =>
      Array(o)
    case In(i, typ) =>
      none
    case Die(message, typ, errorId) =>
      Array(message)
    case Trap(child) => Array(child)
    case ConsoleLog(message, result) =>
      Array(message, result)
    case ApplyIR(_, _, args, _) =>
      args.toFastSeq
    case Apply(_, _, args, _, _) =>
      args.toFastSeq
    case ApplySeeded(_, args, rngState, _, _) =>
      args.toFastSeq :+ rngState
    case ApplySpecial(_, _, args, _, _) =>
      args.toFastSeq
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
    case BlockMatrixMultiWrite(blockMatrices, _) => blockMatrices
    case CollectDistributedArray(ctxs, globals, _, _, body, dynamicID, _, _) => Array(ctxs, globals, body, dynamicID)
    case ReadPartition(path, _, _) => Array(path)
    case WritePartition(stream, ctx, _) => Array(stream, ctx)
    case WriteMetadata(writeAnnotations, _) => Array(writeAnnotations)
    case ReadValue(path, _, _) => Array(path)
    case WriteValue(value, path, _, staged) => Array(value, path) ++ staged.toArray[IR]
    case LiftMeOut(child) => Array(child)
  }
}
