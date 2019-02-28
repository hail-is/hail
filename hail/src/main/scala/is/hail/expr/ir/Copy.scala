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
      case NA(t) => NA(t)
      case IsNA(value) =>
        val IndexedSeq(value: IR) = newChildren
        IsNA(value)
      case If(_, _, _) =>
        val IndexedSeq(cond: IR, cnsq: IR, altr: IR) = newChildren
        If(cond, cnsq, altr)
      case Let(name, _, _) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        Let(name, value, body)
      case AggLet(name, _, _) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        AggLet(name, value, body)
      case Ref(name, t) => Ref(name, t)
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
      case ArrayRef(_, _) =>
        val IndexedSeq(a: IR, i: IR) = newChildren
        ArrayRef(a, i)
      case ArrayLen(_) =>
        val IndexedSeq(a: IR) = newChildren
        ArrayLen(a)
      case ArrayRange(_, _, _) =>
        val IndexedSeq(start: IR, stop: IR, step: IR) = newChildren
        ArrayRange(start, stop, step)
      case MakeNDArray(_, _, _) =>
        val IndexedSeq(data: IR, shape: IR, row_major: IR) = newChildren
        MakeNDArray(data, shape, row_major)
      case NDArrayRef(_, _) =>
        val IndexedSeq(nd: IR, idxs: IR) = newChildren
        NDArrayRef(nd, idxs)
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
      case AggFilter(_, _) =>
        val IndexedSeq(cond: IR, aggIR: IR) = newChildren
        AggFilter(cond, aggIR)
      case AggExplode(_, name, _) =>
        val IndexedSeq(array: IR, aggBody: IR) = newChildren
        AggExplode(array, name, aggBody)
      case AggGroupBy(_, _) =>
        val IndexedSeq(key: IR, aggIR: IR) = newChildren
        AggGroupBy(key, aggIR)
      case AggArrayPerElement(a, name, aggBody) =>
        val IndexedSeq(newA: IR, newAggBody: IR) = newChildren
        AggArrayPerElement(newA, name, newAggBody)
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
      case MakeTuple(_) =>
        MakeTuple(newChildren.map(_.asInstanceOf[IR]))
      case GetTupleElement(_, idx) =>
        val IndexedSeq(o: IR) = newChildren
        GetTupleElement(o, idx)
      case StringSlice(_, _, _) =>
        val IndexedSeq(s: IR, start: IR, n: IR) = newChildren
        StringSlice(s, start, n)
      case StringLength(_) =>
        val IndexedSeq(s: IR) = newChildren
        StringLength(s)
      case In(i, t) => In(i, t)
      case Die(_, typ) =>
        val IndexedSeq(s: IR) = newChildren
        Die(s, typ)
      case x@ApplyIR(fn, args) =>
        val r = ApplyIR(fn, newChildren.map(_.asInstanceOf[IR]))
        r.conversion = x.conversion
        r
      case Apply(fn, args) =>
        Apply(fn, newChildren.map(_.asInstanceOf[IR]))
      case ApplySeeded(fn, args, seed) =>
        ApplySeeded(fn, newChildren.map(_.asInstanceOf[IR]), seed)
      case ApplySpecial(fn, args) =>
        ApplySpecial(fn, newChildren.map(_.asInstanceOf[IR]))
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
      case TableWrite(_, path, overwrite, stageLocally, codecSpecJSONStr) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableWrite(child, path, overwrite, stageLocally, codecSpecJSONStr)
      case TableExport(_, path, typesFile, header, exportType, delimiter) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableExport(child, path, typesFile, header, exportType, delimiter)
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
      case CollectDistributedArray(_, _, cname, gname, _) =>
        val IndexedSeq(ctxs: IR, globals: IR, newBody: IR) = newChildren
        CollectDistributedArray(ctxs, globals, cname, gname, newBody)
    }
  }
}
