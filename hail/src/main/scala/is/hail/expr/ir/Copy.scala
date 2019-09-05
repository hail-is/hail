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
        val IndexedSeq(v: IR) = newChildren
        Cast(v, typ)
      case CastRename(_, typ) =>
        val IndexedSeq(v: IR) = newChildren
        CastRename(v, typ)
      case NA(t) => NA(t)
      case IsNA(value) =>
        val IndexedSeq(value: IR) = newChildren
        IsNA(value)
      case Coalesce(_) =>
        Coalesce(newChildren.map(_.asInstanceOf[IR]))
      case If(_, _, _) =>
        val IndexedSeq(cond: IR, cnsq: IR, altr: IR) = newChildren
        If(cond, cnsq, altr)
      case Let(name, _, _) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        Let(name, value, body)
      case AggLet(name, _, _, isScan) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        AggLet(name, value, body, isScan)
      case Ref(name, t) => Ref(name, t)
      case RelationalRef(name, t) => RelationalRef(name, t)
      case RelationalLet(name, _, _) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        RelationalLet(name, value, body)
      case ApplyBinaryPrimOp(op, _, _) =>
        val IndexedSeq(l: IR, r: IR) = newChildren
        ApplyBinaryPrimOp(op, l, r)
      case ApplyUnaryPrimOp(op, _) =>
        val IndexedSeq(x: IR) = newChildren
        ApplyUnaryPrimOp(op, x)
      case ApplyComparisonOp(op, _, _) =>
        val IndexedSeq(l: IR, r: IR) = newChildren
        ApplyComparisonOp(op, l, r)
      case MakeArray(args, typ) =>
        assert(args.length == newChildren.length)
        MakeArray(newChildren.map(_.asInstanceOf[IR]), typ)
      case MakeStream(args, typ) => 
        assert(args.length == newChildren.length)
        MakeStream(newChildren.map(_.asInstanceOf[IR]), typ)
      case ArrayRef(_, _) =>
        val IndexedSeq(a: IR, i: IR) = newChildren
        ArrayRef(a, i)
      case ArrayLen(_) =>
        val IndexedSeq(a: IR) = newChildren
        ArrayLen(a)
      case ArrayRange(_, _, _) =>
        val IndexedSeq(start: IR, stop: IR, step: IR) = newChildren
        ArrayRange(start, stop, step)
      case StreamRange(_, _, _) =>
        val IndexedSeq(start: IR, stop: IR, step: IR) = newChildren
        StreamRange(start, stop, step)
      case MakeNDArray(_, _, _) =>
        val IndexedSeq(data: IR, shape: IR, rowMajor: IR) = newChildren
        MakeNDArray(data, shape, rowMajor)
      case NDArrayShape(_) =>
        NDArrayShape(newChildren(0).asInstanceOf[IR])
      case NDArrayReshape(_, _) =>
        val IndexedSeq(nd: IR, shape: IR) = newChildren
        NDArrayReshape(nd, shape)
      case NDArrayRef(_, _) =>
        val (nd: IR) +: (idxs: IndexedSeq[_]) = newChildren
        NDArrayRef(nd, idxs.asInstanceOf[IndexedSeq[IR]])
      case NDArraySlice(_, _) =>
        val IndexedSeq(nd: IR, slices: IR) = newChildren
        NDArraySlice(nd, slices)
      case NDArrayMap(_, name, _) =>
        val IndexedSeq(nd: IR, body: IR) = newChildren
        NDArrayMap(nd, name, body)
      case NDArrayMap2(_, _, lName, rName, _) =>
        val IndexedSeq(l: IR, r: IR, body: IR) = newChildren
        NDArrayMap2(l, r, lName, rName, body)
      case NDArrayReindex(_, indexExpr) =>
        val IndexedSeq(nd: IR) = newChildren
        NDArrayReindex(nd, indexExpr)
      case NDArrayAgg(_, axes) =>
        val IndexedSeq(nd: IR) = newChildren
        NDArrayAgg(nd, axes)
      case NDArrayMatMul(_, _) =>
        val IndexedSeq(l: IR, r: IR) = newChildren
        NDArrayMatMul(l, r)
      case NDArrayWrite(_, _) =>
        val IndexedSeq(nd: IR, path: IR) = newChildren
        NDArrayWrite(nd, path)
      case ArraySort(_, l, r, _) =>
        val IndexedSeq(a: IR, comp: IR) = newChildren
        ArraySort(a, l, r, comp)
      case ToSet(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToSet(a)
      case ToDict(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToDict(a)
      case ToArray(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToArray(a)
      case ToStream(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToStream(a)
      case LowerBoundOnOrderedCollection(_, _, asKey) =>
        val IndexedSeq(orderedCollection: IR, elem: IR) = newChildren
        LowerBoundOnOrderedCollection(orderedCollection, elem, asKey)
      case GroupByKey(_) =>
        val IndexedSeq(collection: IR) = newChildren
        GroupByKey(collection)
      case ArrayMap(_, name, _) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        ArrayMap(a, name, body)
      case ArrayFilter(_, name, _) =>
        val IndexedSeq(a: IR, cond: IR) = newChildren
        ArrayFilter(a, name, cond)
      case ArrayFlatMap(_, name, _) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        ArrayFlatMap(a, name, body)
      case ArrayFold(_, _, accumName, valueName, _) =>
        val IndexedSeq(a: IR, zero: IR, body: IR) = newChildren
        ArrayFold(a, zero, accumName, valueName, body)
      case ArrayScan(_, _, accumName, valueName, _) =>
        val IndexedSeq(a: IR, zero: IR, body: IR) = newChildren
        ArrayScan(a, zero, accumName, valueName, body)
      case ArrayLeftJoinDistinct(_, _, l, r, _, _) =>
        val IndexedSeq(left: IR, right: IR, compare: IR, join: IR) = newChildren
        ArrayLeftJoinDistinct(left, right, l, r, compare, join)
      case ArrayFor(_, valueName, _) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        ArrayFor(a, valueName, body)
      case ArrayAgg(_, name, _) =>
        val IndexedSeq(a: IR, query: IR) = newChildren
        ArrayAgg(a, name, query)
      case ArrayAggScan(_, name, _) =>
        val IndexedSeq(a: IR, query: IR) = newChildren
        ArrayAggScan(a, name, query)
      case AggFilter(_, _, isScan) =>
        val IndexedSeq(cond: IR, aggIR: IR) = newChildren
        AggFilter(cond, aggIR, isScan)
      case AggExplode(_, name, _, isScan) =>
        val IndexedSeq(array: IR, aggBody: IR) = newChildren
        AggExplode(array, name, aggBody, isScan)
      case AggGroupBy(_, _, isScan) =>
        val IndexedSeq(key: IR, aggIR: IR) = newChildren
        AggGroupBy(key, aggIR, isScan)
      case AggArrayPerElement(_, elementName, indexName, _, _, isScan) =>
        val (newA, newAggBody, newKnownLength) = newChildren match {
          case IndexedSeq(newA: IR, newAggBody: IR) => (newA, newAggBody, None)
          case IndexedSeq(newA: IR, newAggBody: IR, newKnownLength: IR) => (newA, newAggBody, Some(newKnownLength))
        }
        AggArrayPerElement(newA, elementName, indexName, newAggBody, newKnownLength, isScan)
      case MakeStruct(fields) =>
        assert(fields.length == newChildren.length)
        MakeStruct(fields.zip(newChildren).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) })
      case SelectFields(_, fields) =>
        val IndexedSeq(old: IR) = newChildren
        SelectFields(old, fields)
      case InsertFields(_, fields, fieldOrder) =>
        assert(newChildren.length == fields.length + 1)
        InsertFields(newChildren.head.asInstanceOf[IR], fields.zip(newChildren.tail).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) }, fieldOrder)
      case GetField(_, name) =>
        val IndexedSeq(o: IR) = newChildren
        GetField(o, name)
      case InitOp(_, _, aggSig) =>
        InitOp(newChildren.head.asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]), aggSig)
      case SeqOp(_, _, aggSig) =>
        SeqOp(newChildren.head.asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]), aggSig)
      case InitOp2(i, _, aggSig) =>
        InitOp2(i, newChildren.map(_.asInstanceOf[IR]), aggSig)
      case SeqOp2(i, _, aggSig) =>
        SeqOp2(i, newChildren.map(_.asInstanceOf[IR]), aggSig)
      case x@(_: ResultOp2 | _: CombOp2) =>
        assert(newChildren.isEmpty)
        x
      case x: SerializeAggs => x
      case x: DeserializeAggs => x
      case Begin(_) =>
        Begin(newChildren.map(_.asInstanceOf[IR]))
      case x@ApplyAggOp(_, initOpArgs, _, aggSig) =>
        val args = newChildren.map(_.asInstanceOf[IR])
        ApplyAggOp(
          args.take(x.nConstructorArgs),
          initOpArgs.map(_ => args.drop(x.nConstructorArgs).dropRight(x.nSeqOpArgs)),
          args.takeRight(x.nSeqOpArgs),
          aggSig)
      case x@ApplyScanOp(_, initOpArgs, _, aggSig) =>
        val args = newChildren.map(_.asInstanceOf[IR])
        ApplyScanOp(
          args.take(x.nConstructorArgs),
          initOpArgs.map(_ => args.drop(x.nConstructorArgs).dropRight(x.nSeqOpArgs)),
          args.takeRight(x.nSeqOpArgs),
          aggSig)
      case MakeTuple(fields) =>
        assert(fields.length == newChildren.length)
        MakeTuple(fields.zip(newChildren).map { case ((i, _), newValue) => (i, newValue.asInstanceOf[IR]) })
      case GetTupleElement(_, idx) =>
        val IndexedSeq(o: IR) = newChildren
        GetTupleElement(o, idx)
      case In(i, t) => In(i, t)
      case Die(_, typ) =>
        val IndexedSeq(s: IR) = newChildren
        Die(s, typ)
      case x@ApplyIR(fn, args) =>
        val r = ApplyIR(fn, newChildren.map(_.asInstanceOf[IR]))
        r.conversion = x.conversion
        r
      case Apply(fn, args, t) =>
        Apply(fn, newChildren.map(_.asInstanceOf[IR]), t)
      case ApplySeeded(fn, args, seed, t) =>
        ApplySeeded(fn, newChildren.map(_.asInstanceOf[IR]), seed, t)
      case ApplySpecial(fn, args, t) =>
        ApplySpecial(fn, newChildren.map(_.asInstanceOf[IR]), t)
      case Uniroot(argname, _, _, _) =>
        val IndexedSeq(fn: IR, min: IR, max: IR) = newChildren
        Uniroot(argname, fn, min, max)
      // from MatrixIR
      case MatrixWrite(_, writer) =>
        val IndexedSeq(child: MatrixIR) = newChildren
        MatrixWrite(child, writer)
      case MatrixMultiWrite(_, writer) =>
        MatrixMultiWrite(newChildren.map(_.asInstanceOf[MatrixIR]), writer)
      // from TableIR
      case TableCount(_) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableCount(child)
      case TableGetGlobals(_) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableGetGlobals(child)
      case TableCollect(_) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableCollect(child)
      case TableAggregate(_, _) =>
        val IndexedSeq(child: TableIR, query: IR) = newChildren
        TableAggregate(child, query)
      case MatrixAggregate(_, _) =>
        val IndexedSeq(child: MatrixIR, query: IR) = newChildren
        MatrixAggregate(child, query)
      case TableWrite(_, writer) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableWrite(child, writer)
      case TableMultiWrite(_, writer) =>
        TableMultiWrite(newChildren.map(_.asInstanceOf[TableIR]), writer)
      case TableToValueApply(_, function) =>
        val IndexedSeq(newChild: TableIR) = newChildren
        TableToValueApply(newChild, function)
      case MatrixToValueApply(_, function) =>
        val IndexedSeq(newChild: MatrixIR) = newChildren
        MatrixToValueApply(newChild, function)
      case BlockMatrixToValueApply(_, function) =>
        val IndexedSeq(newChild: BlockMatrixIR) = newChildren
        BlockMatrixToValueApply(newChild, function)
      case BlockMatrixWrite(_, writer) =>
        val IndexedSeq(newChild: BlockMatrixIR) = newChildren
        BlockMatrixWrite(newChild, writer)
      case BlockMatrixMultiWrite(_, writer) =>
        BlockMatrixMultiWrite(newChildren.map(_.asInstanceOf[BlockMatrixIR]), writer)
      case CollectDistributedArray(_, _, cname, gname, _) =>
        val IndexedSeq(ctxs: IR, globals: IR, newBody: IR) = newChildren
        CollectDistributedArray(ctxs, globals, cname, gname, newBody)
      case ReadPartition(path, spec, rowType) =>
        val IndexedSeq(newPath: IR) = newChildren
        ReadPartition(newPath, spec, rowType)
    }
  }
}
