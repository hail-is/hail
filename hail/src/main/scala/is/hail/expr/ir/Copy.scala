package is.hail.expr.ir

object Copy {
  def apply(x: IR, newChildren: IndexedSeq[BaseIR]): IR = {
    x match {
      case I32(value) => I32(value)
      case I64(value) => I64(value)
      case F32(value) => F32(value)
      case F64(value) => F64(value)
      case Str(value) => Str(value)
      case True() => True()
      case False() => False()
      case Literal(typ, value) => Literal(typ, value)
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
      case MakeStream(args, typ) =>
        assert(args.length == newChildren.length)
        MakeStream(newChildren.map(_.asInstanceOf[IR]), typ)
      case ArrayRef(_, _, _) =>
        assert(newChildren.length == 3)
        ArrayRef(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR])
      case ArrayLen(_) =>
        assert(newChildren.length == 1)
        ArrayLen(newChildren(0).asInstanceOf[IR])
      case StreamRange(_, _, _) =>
        assert(newChildren.length == 3)
        StreamRange(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR])
      case ArrayZeros(_) =>
        assert(newChildren.length == 1)
        ArrayZeros(newChildren(0).asInstanceOf[IR])
      case MakeNDArray(_, _, _) =>
        assert(newChildren.length == 3)
        MakeNDArray(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], newChildren(2).asInstanceOf[IR])
      case NDArrayShape(_) =>
        assert(newChildren.length == 1)
        NDArrayShape(newChildren(0).asInstanceOf[IR])
      case NDArrayReshape(_, _) =>
        assert(newChildren.length ==  2)
        NDArrayReshape(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case NDArrayConcat(_, axis) =>
        assert(newChildren.length ==  1)
        NDArrayConcat(newChildren(0).asInstanceOf[IR], axis)
      case NDArrayRef(_, _) =>
        NDArrayRef(newChildren(0).asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]))
      case NDArraySlice(_, _) =>
        assert(newChildren.length ==  2)
        NDArraySlice(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case NDArrayFilter(_, _) =>
        NDArrayFilter(newChildren(0).asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]))
      case NDArrayMap(_, name, _) =>
        assert(newChildren.length ==  2)
        NDArrayMap(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case NDArrayMap2(_, _, lName, rName, _) =>
        assert(newChildren.length ==  3)
        NDArrayMap2(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], lName, rName, newChildren(2).asInstanceOf[IR])
      case NDArrayReindex(_, indexExpr) =>
        assert(newChildren.length == 1)
        NDArrayReindex(newChildren(0).asInstanceOf[IR], indexExpr)
      case NDArrayAgg(_, axes) =>
        assert(newChildren.length == 1)
        NDArrayAgg(newChildren(0).asInstanceOf[IR], axes)
      case NDArrayMatMul(_, _) =>
        assert(newChildren.length == 2)
        NDArrayMatMul(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case NDArrayQR(_, mode) =>
        assert(newChildren.length == 1)
        NDArrayQR(newChildren(0).asInstanceOf[IR], mode)
      case NDArrayWrite(_, _) =>
        assert(newChildren.length == 2)
        NDArrayWrite(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR])
      case ArraySort(_, l, r, _) =>
        assert(newChildren.length == 2)
        ArraySort(newChildren(0).asInstanceOf[IR], l, r, newChildren(1).asInstanceOf[IR])
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
      case ToStream(_) =>
        assert(newChildren.length == 1)
        ToStream(newChildren(0).asInstanceOf[IR])
      case LowerBoundOnOrderedCollection(_, _, asKey) =>
        assert(newChildren.length == 2)
        LowerBoundOnOrderedCollection(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], asKey)
      case GroupByKey(_) =>
        assert(newChildren.length == 1)
        GroupByKey(newChildren(0).asInstanceOf[IR])
      case StreamMap(_, name, _) =>
        assert(newChildren.length == 2)
        StreamMap(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
      case StreamZip(_, names, _, behavior) =>
        assert(newChildren.length == names.length + 1)
        StreamZip(newChildren.init.asInstanceOf[IndexedSeq[IR]], names, newChildren(names.length).asInstanceOf[IR], behavior)
      case StreamFilter(_, name, _) =>
        assert(newChildren.length == 2)
        StreamFilter(newChildren(0).asInstanceOf[IR], name, newChildren(1).asInstanceOf[IR])
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
      case StreamLeftJoinDistinct(_, _, l, r, _, _) =>
        assert(newChildren.length == 4)
        StreamLeftJoinDistinct(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], l, r, newChildren(2).asInstanceOf[IR], newChildren(3).asInstanceOf[IR])
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
      case InitOp(i, _, aggSig, op) =>
        InitOp(i, newChildren.map(_.asInstanceOf[IR]), aggSig, op)
      case SeqOp(i, _, aggSig, op) =>
        SeqOp(i, newChildren.map(_.asInstanceOf[IR]), aggSig, op)
      case x@(_: ResultOp | _: CombOp | _: AggStateValue) =>
        assert(newChildren.isEmpty)
        x
      case CombOpValue(i, _, aggSig) =>
        assert(newChildren.length == 1)
        CombOpValue(i, newChildren(0).asInstanceOf[IR], aggSig)
      case x: SerializeAggs => x
      case x: DeserializeAggs => x
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
      case MakeTuple(fields) =>
        assert(fields.length == newChildren.length)
        MakeTuple(fields.zip(newChildren).map { case ((i, _), newValue) => (i, newValue.asInstanceOf[IR]) })
      case GetTupleElement(_, idx) =>
        assert(newChildren.length == 1)
        GetTupleElement(newChildren(0).asInstanceOf[IR], idx)
      case In(i, t) => In(i, t)
      case Die(_, typ) =>
        assert(newChildren.length == 1)
        Die(newChildren(0).asInstanceOf[IR], typ)
      case x@ApplyIR(fn, args) =>
        val r = ApplyIR(fn, newChildren.map(_.asInstanceOf[IR]))
        r.conversion = x.conversion
        r.inline = x.inline
        r
      case Apply(fn, args, t) =>
        Apply(fn, newChildren.map(_.asInstanceOf[IR]), t)
      case ApplySeeded(fn, args, seed, t) =>
        ApplySeeded(fn, newChildren.map(_.asInstanceOf[IR]), seed, t)
      case ApplySpecial(fn, args, t) =>
        ApplySpecial(fn, newChildren.map(_.asInstanceOf[IR]), t)
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
      case UnpersistBlockMatrix(child) =>
        assert(newChildren.length == 1)
        UnpersistBlockMatrix(newChildren(0).asInstanceOf[BlockMatrixIR])
      case CollectDistributedArray(_, _, cname, gname, _) =>
        assert(newChildren.length == 3)
        CollectDistributedArray(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], cname, gname, newChildren(2).asInstanceOf[IR])
      case ReadPartition(path, spec, rowType) =>
        assert(newChildren.length == 1)
        ReadPartition(newChildren(0).asInstanceOf[IR], spec, rowType)
      case ReadValue(path, spec, requestedType) =>
        assert(newChildren.length == 1)
        ReadValue(newChildren(0).asInstanceOf[IR], spec, requestedType)
      case WriteValue(value, pathPrefix, spec) =>
        assert(newChildren.length == 2)
        WriteValue(newChildren(0).asInstanceOf[IR], newChildren(1).asInstanceOf[IR], spec)
      case LiftMeOut(_) =>
        LiftMeOut(newChildren(0).asInstanceOf[IR])
    }
  }
}
