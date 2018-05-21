package is.hail.expr.ir

import is.hail.expr.{BaseIR, MatrixIR, TableIR}

object Copy {
  def apply(x: IR, newChildren: IndexedSeq[BaseIR]): BaseIR = {
    lazy val same = {
      assert(newChildren.isEmpty)
      x
    }
    x match {
      case I32(_) => same
      case I64(_) => same
      case F32(_) => same
      case F64(_) => same
      case Str(_) => same
      case True() => same
      case False() => same
      case Void() => same
      case Cast(_, typ) =>
        val IndexedSeq(v: IR) = newChildren
        Cast(v, typ)
      case NA(_) => same
      case IsNA(value) =>
        val IndexedSeq(value: IR) = newChildren
        IsNA(value)
      case If(_, _, _) =>
        val IndexedSeq(cond: IR, cnsq: IR, altr: IR) = newChildren
        If(cond, cnsq, altr)
      case Let(name, _, _) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        Let(name, value, body)
      case Ref(_, _) =>
        same
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
      case ArraySort(_) =>
        val IndexedSeq(a: IR) = newChildren
        ArraySort(a)
      case ToSet(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToSet(a)
      case ToDict(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToDict(a)
      case ToArray(_) =>
        val IndexedSeq(a: IR) = newChildren
        ToArray(a)
      case SetContains(_, _) =>
        val IndexedSeq(set: IR, elem: IR) = newChildren
        SetContains(set, elem)
      case DictContains(_, _) =>
        val IndexedSeq(dict: IR, key: IR) = newChildren
        DictContains(dict, key)
      case DictGet(_, _) =>
        val IndexedSeq(dict: IR, key: IR) = newChildren
        DictGet(dict, key)
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
      case ArrayFor(_, valueName, _) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        ArrayFor(a, valueName, body)
      case MakeStruct(fields) =>
        assert(fields.length == newChildren.length)
        MakeStruct(fields.zip(newChildren).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) })
      case SelectFields(_, fields) =>
        val IndexedSeq(old: IR) = newChildren
        SelectFields(old, fields)
      case InsertFields(_, fields) =>
        assert(newChildren.length == fields.length + 1)
        InsertFields(newChildren.head.asInstanceOf[IR], fields.zip(newChildren.tail).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) })
      case GetField(_, name) =>
        val IndexedSeq(o: IR) = newChildren
        GetField(o, name)
      case InitOp(_, _, aggSig) =>
        InitOp(newChildren.head.asInstanceOf[IR], newChildren.tail.map(_.asInstanceOf[IR]), aggSig)
      case SeqOp(_, _, aggSig) =>
        val IndexedSeq(a: IR, i: IR) = newChildren
        SeqOp(a, i, aggSig)
      case Begin(_) =>
        Begin(newChildren.map(_.asInstanceOf[IR]))
      case x@ApplyAggOp(_, _, initOpArgs, aggSig) =>
        val args = newChildren.map(_.asInstanceOf[IR])
        ApplyAggOp(
          args.head,
          args.tail.take(x.nConstructorArgs),
          initOpArgs.map(_ => args.drop(x.nConstructorArgs + 1)),
          aggSig)
      case MakeTuple(_) =>
        MakeTuple(newChildren.map(_.asInstanceOf[IR]))
      case GetTupleElement(_, idx) =>
        val IndexedSeq(o: IR) = newChildren
        GetTupleElement(o, idx)
      case In(_, _) =>
        same
      case Die(message) =>
        same
      case ApplyIR(fn, args, conversion) =>
        ApplyIR(fn, newChildren.map(_.asInstanceOf[IR]), conversion)
      case Apply(fn, args) =>
        Apply(fn, newChildren.map(_.asInstanceOf[IR]))
      case ApplySpecial(fn, args) =>
        ApplySpecial(fn, newChildren.map(_.asInstanceOf[IR]))
      // from MatrixIR
      case MatrixWrite(_, f) =>
        val IndexedSeq(child: MatrixIR) = newChildren
        MatrixWrite(child, f)
      // from TableIR
      case TableCount(_) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableCount(child)
      case TableAggregate(_, _) =>
        val IndexedSeq(child: TableIR, query: IR) = newChildren
        TableAggregate(child, query)
      case TableWrite(_, path, overwrite, codecSpecJSONStr) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableWrite(child, path, overwrite, codecSpecJSONStr)
      case TableExport(_, path, typesFile, header, exportType) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableExport(child, path, typesFile, header, exportType)
    }
  }
}
