package is.hail.expr.ir

object Copy {
  def apply(x: IR, newChildren: IndexedSeq[BaseIR]): IR = {
    x match {
      case I32(value) => I32(value)
      case I64(value) => I64(value)
      case F32(value) => F32(value)
      case F64(value) => F64(value)
      case Str(value) => Str(value)
      case UUID4(id) => UUID4(id)
      case True() => True()
      case False() => False()
      case Literal(typ, value) => Literal(typ, value)
      case EncodedLiteral(codec, value) => EncodedLiteral(codec, value)
      case Void() => Void()
      case Cast(_, typ) =>
        assert(newChildren.length == 1)
        Cast(newChildren(0).asInstanceOf[IR], typ)
      case CastRename(_, typ) =>
        assert(newChildren.length == 1)
        CastRename(newChildren(0).asInstanceOf[IR], typ)
      case NA(t) => NA(t)
      case IsNA(value) =>
        assert(newChildren.length == 1)
        IsNA(newChildren(0).asInstanceOf[IR])
      case Coalesce(_) =>
        Coalesce(newChildren.map(_.asInstanceOf[IR]))
      case Consume(_) =>
        Consume(newChildren(0).asInstanceOf[IR])
      case If(_, _, _) =>
        assert(newChildren.length == 3)
        If(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR])
      case Let(name, _, _) =>
        assert(newChildren.length == 2)
        Let(name, newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case AggLet(name, _, _, isScan) =>
        assert(newChildren.length == 2)
        AggLet(name, newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], isScan)
      case TailLoop(name, params, _) =>
        assert(newChildren.length == params.length + 1)
        TailLoop(name, params.map(_._1).zip(newChildren.init.map(_.asInstanceOf[IR])), newChildren.last.asInstanceOf[IR])
      case Recur(name, args, t) =>
        assert(newChildren.length == args.length)
        Recur(name, newChildren.map(_.asInstanceOf[IR]), t)
      case Ref(name, t) => Ref(name, t)
      case RelationalRef(name, t) => RelationalRef(name, t)
      case RelationalLet(name, _, _) =>
        assert(newChildren.length == 2)
        RelationalLet(name, newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case ApplyBinaryPrimOp(op, _, _) =>
        assert(newChildren.length == 2)
        ApplyBinaryPrimOp(op, newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case ApplyUnaryPrimOp(op, _) =>
        assert(newChildren.length == 1)
        ApplyUnaryPrimOp(op, newChildren(0).asInstanceOf[IR])
      case ApplyComparisonOp(op, _, _) =>
        assert(newChildren.length == 2)
        ApplyComparisonOp(op, newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case MakeArray(args, typ) =>
        assert(args.length == newChildren.length)
        MakeArray(newChildren.map(_.asInstanceOf[IR]), typ)
      case MakeStream(args, typ, requiresMemoryManagementPerElement) =>
        assert(args.length == newChildren.length)
        MakeStream(newChildren.map(_.asInstanceOf[IR]), typ, requiresMemoryManagementPerElement)
      case ArrayRef(_, _, errorID) =>
        assert(newChildren.length == 2)
        ArrayRef(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], errorID)
      case ArraySlice(_,_, stop, _, errorID) =>
        if (stop.isEmpty) {
          assert(newChildren.length == 3)
          ArraySlice(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], None,
            newChildren(2).asInstanceOf[IR], errorID)
        }
        else {
            assert(newChildren.length == 4)
            ArraySlice(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], Some(newChildren(2).asInstanceOf[IR]),
              newChildren(3).asInstanceOf[IR], errorID)
          }
      case ArrayLen(_) =>
        assert(newChildren.length == 1)
        ArrayLen(newChildren(0).asInstanceOf[IR])
      case StreamIota(_, _, requiresMemoryManagementPerElement) =>
        assert(newChildren.length == 2)
        StreamIota(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], requiresMemoryManagementPerElement)
      case StreamRange(_, _, _, requiresMemoryManagementPerElement, errorID) =>
        assert(newChildren.length == 3)
        StreamRange(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR],
          requiresMemoryManagementPerElement, errorID)
      case SeqSample(_, _, _, requiresMemoryManagementPerElement) =>
        assert(newChildren.length == 3)
        SeqSample(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR], requiresMemoryManagementPerElement)
      case StreamDistribute(_, _, _, op, spec) =>
        StreamDistribute(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR], op, spec)
      case StreamWhiten(stream, newChunk, prevWindow, vecSize, windowSize, chunkSize, blockSize, normalizeAfterWhiten) =>
        StreamWhiten(newChildren(0).asInstanceOf[IR], newChunk, prevWindow, vecSize, windowSize, chunkSize, blockSize, normalizeAfterWhiten)
      case ArrayZeros(_) =>
        assert(newChildren.length == 1)
        ArrayZeros(newChildren(0).asInstanceOf[IR])
      case MakeNDArray(_, _, _, errorId) =>
        assert(newChildren.length == 3)
        MakeNDArray(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR], errorId)
      case NDArrayShape(_) =>
        assert(newChildren.length == 1)
        NDArrayShape(newChildren(0).asInstanceOf[IR])
      case NDArrayReshape(_, _, errorID) =>
        assert(newChildren.length ==  2)
        NDArrayReshape(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], errorID)
      case NDArrayConcat(_, axis) =>
        assert(newChildren.length ==  1)
        NDArrayConcat(newChildren(0).asInstanceOf[IR], axis)
      case NDArrayRef(_, _, errorId) =>
        NDArrayRef(newChildren(0).asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]), errorId)
      case NDArraySlice(_, _) =>
        assert(newChildren.length ==  2)
        NDArraySlice(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case NDArrayFilter(_, _) =>
        NDArrayFilter(newChildren(0).asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]))
      case NDArrayMap(_, name, _) =>
        assert(newChildren.length ==  2)
        NDArrayMap(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case NDArrayMap2(_, _, lName, rName, _, errorID) =>
        assert(newChildren.length ==  3)
        NDArrayMap2(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], lName, rName, newChildren(2).asInstanceOf[IR], errorID)
      case NDArrayReindex(_, indexExpr) =>
        assert(newChildren.length == 1)
        NDArrayReindex(newChildren(0).asInstanceOf[IR], indexExpr)
      case NDArrayAgg(_, axes) =>
        assert(newChildren.length == 1)
        NDArrayAgg(newChildren(0).asInstanceOf[IR], axes)
      case NDArrayMatMul(_, _, errorID) =>
        assert(newChildren.length == 2)
        NDArrayMatMul(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], errorID)
      case NDArrayQR(_, mode, errorID) =>
        assert(newChildren.length == 1)
        NDArrayQR(newChildren(0).asInstanceOf[IR], mode, errorID)
      case NDArraySVD(_, fullMatrices, computeUV, errorID) =>
        assert(newChildren.length == 1)
        NDArraySVD(newChildren(0).asInstanceOf[IR], fullMatrices, computeUV, errorID)
      case NDArrayInv(_, errorID) =>
        assert(newChildren.length == 1)
        NDArrayInv(newChildren(0).asInstanceOf[IR], errorID)
      case NDArrayWrite(_, _) =>
        assert(newChildren.length == 2)
        NDArrayWrite(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case ArraySort(_, l, r, _) =>
        assert(newChildren.length == 2)
        ArraySort(newChildren(0).asInstanceOf[IR], l, r, newChildren(1).asInstanceOf[IR])
      case ArrayMaximalIndependentSet(_, tb) =>
        ArrayMaximalIndependentSet(newChildren(0).asInstanceOf[IR], tb.map { case (l, r, _) => (l, r, newChildren(1).asInstanceOf[IR]) } )
      case ToSet(_) =>
        assert(newChildren.length == 1)
        ToSet(newChildren(0).asInstanceOf[IR])
      case ToDict(_) =>
        assert(newChildren.length == 1)
        ToDict(newChildren(0).asInstanceOf[IR])
      case ToArray(_) =>
        assert(newChildren.length == 1)
        ToArray(newChildren(0).asInstanceOf[IR])
      case CastToArray(_) =>
        assert(newChildren.length == 1)
        CastToArray(newChildren(0).asInstanceOf[IR])
      case ToStream(_, requiresMemoryManagementPerElement) =>
        assert(newChildren.length == 1)
        ToStream(newChildren(0).asInstanceOf[IR], requiresMemoryManagementPerElement)
      case LowerBoundOnOrderedCollection(_, _, asKey) =>
        assert(newChildren.length == 2)
        LowerBoundOnOrderedCollection(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], asKey)
      case GroupByKey(_) =>
        assert(newChildren.length == 1)
        GroupByKey(newChildren(0).asInstanceOf[IR])
      case RNGStateLiteral() => RNGStateLiteral()
      case RNGSplit(_, _) =>
        assert(newChildren.nonEmpty)
        RNGSplit(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case StreamLen(_) =>
        StreamLen(newChildren(0).asInstanceOf[IR])
      case StreamTake(_, _) =>
        assert(newChildren.length == 2)
        StreamTake(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case StreamDrop(_, _) =>
        assert(newChildren.length == 2)
        StreamDrop(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case StreamGrouped(_, _) =>
        assert(newChildren.length == 2)
        StreamGrouped(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case StreamGroupByKey(_, key, missingEqual) =>
        assert(newChildren.length == 1)
        StreamGroupByKey(newChildren(0).asInstanceOf[IR], key, missingEqual)
      case StreamMap(_, name, _) =>
        assert(newChildren.length == 2)
        StreamMap(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamZip(_, names, _, behavior, errorID) =>
        assert(newChildren.length == names.length + 1)
        StreamZip(newChildren.init.asInstanceOf[IndexedSeq[IR]], names, newChildren(names.length).asInstanceOf[IR],
          behavior, errorID)
      case StreamZipJoin(as, key, curKey, curVals, _) =>
        assert(newChildren.length == as.length + 1)
        StreamZipJoin(newChildren.init.asInstanceOf[IndexedSeq[IR]], key, curKey, curVals, newChildren(as.length).asInstanceOf[IR])
      case StreamMultiMerge(as, key) =>
        assert(newChildren.length == as.length)
        StreamMultiMerge(newChildren.asInstanceOf[IndexedSeq[IR]], key)
      case StreamFilter(_, name, _) =>
        assert(newChildren.length == 2)
        StreamFilter(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamTakeWhile(_, name, _) =>
        assert(newChildren.length == 2)
        StreamTakeWhile(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamDropWhile(_, name, _) =>
        assert(newChildren.length == 2)
        StreamDropWhile(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamFlatMap(_, name, _) =>
        assert(newChildren.length == 2)
        StreamFlatMap(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamFold(_, _, accumName, valueName, _) =>
        assert(newChildren.length == 3)
        StreamFold(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], accumName, valueName, newChildren(2).asInstanceOf[IR])
      case StreamFold2(_, accum, valueName, seq, _) =>
        val ncIR = newChildren.map(_.asInstanceOf[IR])
        assert(newChildren.length == 2 + accum.length + seq.length)
        StreamFold2(ncIR(0),
          accum.indices.map(i => (accum(i)._1, ncIR(i + 1))),
          valueName,
          seq.indices.map(i => ncIR(i + 1 + accum.length)), ncIR.last)
      case StreamScan(_, _, accumName, valueName, _) =>
        assert(newChildren.length == 3)
        StreamScan(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], accumName, valueName, newChildren(2).asInstanceOf[IR])
      case StreamJoinRightDistinct(_, _, lKey, rKey, l, r, _, joinType) =>
        assert(newChildren.length == 3)
        StreamJoinRightDistinct(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], lKey, rKey, l, r, newChildren(2).asInstanceOf[IR], joinType)
      case _: StreamLocalLDPrune =>
        val IndexedSeq(child: IR, r2Threshold: IR, windowSize: IR, maxQueueSize: IR, nSamples: IR) = newChildren
        StreamLocalLDPrune(child, r2Threshold, windowSize, maxQueueSize, nSamples)
      case StreamFor(_, valueName, _) =>
        assert(newChildren.length == 2)
        StreamFor(newChildren(0).asInstanceOf[IR], valueName, newChildren(1).asInstanceOf[IR])
      case StreamAgg(_, name, _) =>
        assert(newChildren.length == 2)
        StreamAgg(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamAggScan(_, name, _) =>
        assert(newChildren.length == 2)
        StreamAggScan(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case RunAgg(_, _, signatures) =>
        RunAgg(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], signatures)
      case RunAggScan(_, name, _, _, _, signatures) =>
        RunAggScan(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR],
          newChildren(2).asInstanceOf[IR], newChildren(3).asInstanceOf[IR], signatures)
      case StreamBufferedAggregate(_, _, _, _, name, aggSignatures, bufferSize) =>
        StreamBufferedAggregate(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR],
          newChildren(2).asInstanceOf[IR], newChildren(3).asInstanceOf[IR], name, aggSignatures, bufferSize)
      case AggFilter(_, _, isScan) =>
        assert(newChildren.length == 2)
        AggFilter(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], isScan)
      case AggExplode(_, name, _, isScan) =>
        assert(newChildren.length == 2)
        AggExplode(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR], isScan)
      case AggGroupBy(_, _, isScan) =>
        assert(newChildren.length == 2)
        AggGroupBy(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], isScan)
      case AggArrayPerElement(_, elementName, indexName, _, _, isScan) =>
        val newKnownLength = if (newChildren.length == 3)
          Some(newChildren(2).asInstanceOf[IR])
        else {
          assert(newChildren.length == 2)
          None
        }
        AggArrayPerElement(newChildren(0).asInstanceOf[IR], elementName, indexName, newChildren(1).asInstanceOf[IR], newKnownLength, isScan)
      case MakeStruct(fields) =>
        assert(fields.length == newChildren.length)
        MakeStruct(fields.zip(newChildren).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) })
      case SelectFields(_, fields) =>
        assert(newChildren.length == 1)
        SelectFields(newChildren(0).asInstanceOf[IR], fields)
      case InsertFields(_, fields, fieldOrder) =>
        assert(newChildren.length == fields.length + 1)
        InsertFields(newChildren.head.asInstanceOf[IR], fields.zip(newChildren.tail).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) }, fieldOrder)
      case GetField(_, name) =>
        assert(newChildren.length == 1)
        GetField(newChildren(0).asInstanceOf[IR], name)
      case InitOp(i, _, aggSig) =>
        InitOp(i, newChildren.map(_.asInstanceOf[IR]), aggSig)
      case SeqOp(i, _, aggSig) =>
        SeqOp(i, newChildren.map(_.asInstanceOf[IR]), aggSig)
      case ResultOp(i, aggSigs) =>
        ResultOp(i, aggSigs)
      case CombOp(i1, i2, aggSig) =>
        CombOp(i1, i2, aggSig)
      case AggStateValue(i, aggSig) =>
        AggStateValue(i, aggSig)
      case CombOpValue(i, _, aggSig) =>
        assert(newChildren.length == 1)
        CombOpValue(i, newChildren(0).asInstanceOf[IR], aggSig)
      case InitFromSerializedValue(i, _, aggSig) =>
        assert(newChildren.length == 1)
        InitFromSerializedValue(i, newChildren(0).asInstanceOf[IR], aggSig)
      case SerializeAggs(startIdx, serIdx, spec, aggSigs) => SerializeAggs(startIdx, serIdx, spec, aggSigs)
      case DeserializeAggs(startIdx, serIdx, spec, aggSigs) => DeserializeAggs(startIdx, serIdx, spec, aggSigs)
      case Begin(_) =>
        Begin(newChildren.map(_.asInstanceOf[IR]))
      case x@ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
        val args = newChildren.map(_.asInstanceOf[IR])
        assert(args.length == x.nInitArgs + x.nSeqOpArgs)
        ApplyAggOp(
          args.take(x.nInitArgs),
          args.drop(x.nInitArgs),
          aggSig)
      case x@ApplyScanOp(initOpArgs, _, aggSig) =>
        val args = newChildren.map(_.asInstanceOf[IR])
        assert(args.length == x.nInitArgs + x.nSeqOpArgs)
        ApplyScanOp(
          args.take(x.nInitArgs),
          args.drop(x.nInitArgs),
          aggSig)
      case AggFold(_, _, _, accumName, otherAccumName, isScan) =>
        AggFold(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR], accumName, otherAccumName, isScan)
      case MakeTuple(fields) =>
        assert(fields.length == newChildren.length)
        MakeTuple(fields.zip(newChildren).map { case ((i, _), newValue) => (i, newValue.asInstanceOf[IR]) })
      case GetTupleElement(_, idx) =>
        assert(newChildren.length == 1)
        GetTupleElement(newChildren(0).asInstanceOf[IR], idx)
      case In(i, t) => In(i, t)
      case Die(_, typ, errorId) =>
        assert(newChildren.length == 1)
        Die(newChildren(0).asInstanceOf[IR], typ, errorId)
      case Trap(child) =>
        assert(newChildren.length == 1)
        Trap(newChildren(0).asInstanceOf[IR])
      case ConsoleLog(message, result) =>
        assert(newChildren.length == 2)
        ConsoleLog(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case x@ApplyIR(fn, typeArgs, args, errorID) =>
        val r = ApplyIR(fn, typeArgs, newChildren.map(_.asInstanceOf[IR]), errorID)
        r.conversion = x.conversion
        r.inline = x.inline
        r
      case Apply(fn, typeArgs, args, t, errorID) =>
        Apply(fn, typeArgs, newChildren.map(_.asInstanceOf[IR]), t, errorID)
      case ApplySeeded(fn, args, rngState, staticUID, t) =>
        ApplySeeded(fn, newChildren.init.map(_.asInstanceOf[IR]), newChildren.last.asInstanceOf[IR], staticUID, t)
      case ApplySpecial(fn, typeArgs, args, t, errorID) =>
        ApplySpecial(fn, typeArgs, newChildren.map(_.asInstanceOf[IR]), t, errorID)
      // from MatrixIR
      case MatrixWrite(_, writer) =>
        assert(newChildren.length == 1)
        MatrixWrite(newChildren(0).asInstanceOf[MatrixIR], writer)
      case MatrixMultiWrite(_, writer) =>
        MatrixMultiWrite(newChildren.map(_.asInstanceOf[MatrixIR]), writer)
      case MatrixCount(_) =>
        assert(newChildren.length == 1)
        MatrixCount(newChildren(0).asInstanceOf[MatrixIR])
      // from TableIR
      case TableCount(_) =>
        assert(newChildren.length == 1)
        TableCount(newChildren(0).asInstanceOf[TableIR])
      case TableGetGlobals(_) =>
        assert(newChildren.length == 1)
        TableGetGlobals(newChildren(0).asInstanceOf[TableIR])
      case TableCollect(_) =>
        assert(newChildren.length == 1)
        TableCollect(newChildren(0).asInstanceOf[TableIR])
      case TableAggregate(_, _) =>
        assert(newChildren.length == 2)
        TableAggregate(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
      case MatrixAggregate(_, _) =>
        assert(newChildren.length == 2)
        MatrixAggregate(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
      case TableWrite(_, writer) =>
        assert(newChildren.length == 1)
        TableWrite(newChildren(0).asInstanceOf[TableIR], writer)
      case TableMultiWrite(_, writer) =>
        TableMultiWrite(newChildren.map(_.asInstanceOf[TableIR]), writer)
      case TableToValueApply(_, function) =>
        assert(newChildren.length == 1)
        TableToValueApply(newChildren(0).asInstanceOf[TableIR], function)
      case MatrixToValueApply(_, function) =>
        assert(newChildren.length == 1)
        MatrixToValueApply(newChildren(0).asInstanceOf[MatrixIR], function)
      case BlockMatrixToValueApply(_, function) =>
        assert(newChildren.length == 1)
        BlockMatrixToValueApply(newChildren(0).asInstanceOf[BlockMatrixIR], function)
      case BlockMatrixCollect(_) =>
        assert(newChildren.length == 1)
        BlockMatrixCollect(newChildren(0).asInstanceOf[BlockMatrixIR])
      case BlockMatrixWrite(_, writer) =>
        assert(newChildren.length == 1)
        BlockMatrixWrite(newChildren(0).asInstanceOf[BlockMatrixIR], writer)
      case BlockMatrixMultiWrite(_, writer) =>
        BlockMatrixMultiWrite(newChildren.map(_.asInstanceOf[BlockMatrixIR]), writer)
      case CollectDistributedArray(_, _, cname, gname, _, _, id, tsd) =>
        assert(newChildren.length == 4)
        CollectDistributedArray(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], cname, gname, newChildren(2).asInstanceOf[IR], newChildren(3).asInstanceOf[IR], id, tsd)
      case ReadPartition(context, rowType, reader) =>
        assert(newChildren.length == 1)
        ReadPartition(newChildren(0).asInstanceOf[IR], rowType, reader)
      case WritePartition(stream, ctx, writer) =>
        assert(newChildren.length == 2)
        WritePartition(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], writer)
      case WriteMetadata(ctx, writer) =>
        assert(newChildren.length == 1)
        WriteMetadata(newChildren(0).asInstanceOf[IR], writer)
      case ReadValue(path, spec, requestedType) =>
        assert(newChildren.length == 1)
        ReadValue(newChildren(0).asInstanceOf[IR], spec, requestedType)
      case WriteValue(_, _, writer, _) =>
        assert(newChildren.length == 2 || newChildren.length == 3)
        val value = newChildren(0).asInstanceOf[IR]
        val path = newChildren(1).asInstanceOf[IR]
        val stage = if (newChildren.length == 3) Some(newChildren(2).asInstanceOf[IR]) else None
        WriteValue(value, path, writer, stage)
      case LiftMeOut(_) =>
        LiftMeOut(newChildren(0).asInstanceOf[IR])
    }
  }
}
