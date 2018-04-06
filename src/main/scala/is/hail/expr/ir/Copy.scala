package is.hail.expr.ir

import is.hail.expr.{BaseIR, TableIR}

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
      case True() => same
      case False() => same
      case Cast(_, typ) =>
        val IndexedSeq(v: IR) = newChildren
        Cast(v, typ)
      case NA(_) => same
      case MapNA(name, _, _, typ) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        MapNA(name, value, body, typ)
      case IsNA(value) =>
        val IndexedSeq(value: IR) = newChildren
        IsNA(value)
      case If(_, _, _, typ) =>
        val IndexedSeq(cond: IR, cnsq: IR, altr: IR) = newChildren
        If(cond, cnsq, altr, typ)
      case Let(name, _, _, typ) =>
        val IndexedSeq(value: IR, body: IR) = newChildren
        Let(name, value, body, typ)
      case Ref(_, _) =>
        same
      case ApplyBinaryPrimOp(op, _, _, typ) =>
        val IndexedSeq(l: IR, r: IR) = newChildren
        ApplyBinaryPrimOp(op, l, r, typ)
      case ApplyUnaryPrimOp(op, _, typ) =>
        val IndexedSeq(x: IR) = newChildren
        ApplyUnaryPrimOp(op, x, typ)
      case MakeArray(args, typ) =>
        assert(args.length == newChildren.length)
        MakeArray(newChildren.map(_.asInstanceOf[IR]), typ)
      case ArrayRef(_, _, typ) =>
        val IndexedSeq(a: IR, i: IR) = newChildren
        ArrayRef(a, i, typ)
      case ArrayMissingnessRef(_, _) =>
        val IndexedSeq(a: IR, i: IR) = newChildren
        ArrayMissingnessRef(a, i)
      case ArrayLen(_) =>
        val IndexedSeq(a: IR) = newChildren
        ArrayLen(a)
      case ArrayRange(_, _, _) =>
        val IndexedSeq(start: IR, stop: IR, step: IR) = newChildren
        ArrayRange(start, stop, step)
      case ArrayMap(_, name, _, elementTyp) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        ArrayMap(a, name, body, elementTyp)
      case ArrayFilter(_, name, _) =>
        val IndexedSeq(a: IR, cond: IR) = newChildren
        ArrayFilter(a, name, cond)
      case ArrayFlatMap(_, name, _) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        ArrayFlatMap(a, name, body)
      case ArrayFold(_, _, accumName, valueName, _, typ) =>
        val IndexedSeq(a: IR, zero: IR, body: IR) = newChildren
        ArrayFold(a, zero, accumName, valueName, body, typ)
      case MakeStruct(fields, _) =>
        assert(fields.length == newChildren.length)
        MakeStruct(fields.zip(newChildren).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) })
      case InsertFields(_, fields, typ) =>
        assert(newChildren.length == fields.length + 1)
        InsertFields(newChildren.head.asInstanceOf[IR], fields.zip(newChildren.tail).map { case ((n, _), a) => (n, a.asInstanceOf[IR]) })
      case GetField(_, name, typ) =>
        val IndexedSeq(o: IR) = newChildren
        GetField(o, name, typ)
      case GetFieldMissingness(_, name) =>
        val IndexedSeq(o: IR) = newChildren
        GetFieldMissingness(o, name)
      case AggIn(_) =>
        same
      case AggMap(_, name, _, typ) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        AggMap(a, name, body, typ)
      case AggFilter(_, name, _, typ) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        AggFilter(a, name, body, typ)
      case AggFlatMap(_, name, _, typ) =>
        val IndexedSeq(a: IR, body: IR) = newChildren
        AggFlatMap(a, name, body, typ)
      case ApplyAggOp(_, op, args, typ) =>
        ApplyAggOp(newChildren.head.asInstanceOf[IR], op, newChildren.tail.map(_.asInstanceOf[IR]), typ)
      case MakeTuple(_, typ) =>
        MakeTuple(newChildren.map(_.asInstanceOf[IR]), typ)
      case GetTupleElement(_, idx, typ) =>
        val IndexedSeq(o: IR) = newChildren
        GetTupleElement(o, idx, typ)
      case In(_, _) =>
        same
      case InMissingness(_) =>
        same
      case Die(message) =>
        same
      case Apply(fn, args, impl) =>
        Apply(fn, newChildren.map(_.asInstanceOf[IR]), impl)
      case ApplySpecial(fn, args, impl) =>
        ApplySpecial(fn, newChildren.map(_.asInstanceOf[IR]), impl)
      // from TableIR
      case TableCount(_) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableCount(child)
      case TableAggregate(_, _, typ) =>
        val IndexedSeq(child: TableIR, query: IR) = newChildren
        TableAggregate(child, query)
      case TableWrite(_, path, overwrite, codecSpecJSONStr) =>
        val IndexedSeq(child: TableIR) = newChildren
        TableWrite(child, path, overwrite, codecSpecJSONStr)
    }
  }
}
