package is.hail.expr.ir

object Recur {
  def apply(f: IR => IR)(ir: IR): IR = ir match {
    case I32(x) => ir
    case I64(x) => ir
    case F32(x) => ir
    case F64(x) => ir
    case True() => ir
    case False() => ir
    case Cast(v, typ) => Cast(f(v), typ)
    case NA(typ) => ir
    case MapNA(name, value, body, typ) => MapNA(name, f(value), f(body), typ)
    case IsNA(value) => IsNA(f(value))
    case If(cond, cnsq, altr, typ) => If(f(cond), f(cnsq), f(altr), typ)
    case Let(name, value, body, typ) => Let(name, f(value), f(body), typ)
    case Ref(name, typ) => ir
    case ApplyBinaryPrimOp(op, l, r, typ) => ApplyBinaryPrimOp(op, f(l), f(r), typ)
    case ApplyUnaryPrimOp(op, x, typ) => ApplyUnaryPrimOp(op, f(x), typ)
    case MakeArray(args, typ) => MakeArray(args map f, typ)
    case ArrayRef(a, i, typ) => ArrayRef(f(a), f(i), typ)
    case ArrayMissingnessRef(a, i) => ArrayMissingnessRef(f(a), f(i))
    case ArrayLen(a) => ArrayLen(f(a))
    case ArrayRange(start, stop, step) => ArrayRange(f(start), f(stop), f(step))
    case ArrayMap(a, name, body, elementTyp) => ArrayMap(f(a), name, f(body), elementTyp)
    case ArrayFilter(a, name, cond) => ArrayFilter(f(a), name, f(cond))
    case ArrayFlatMap(a, name, body) => ArrayFlatMap(f(a), name, f(body))
    case ArrayFold(a, zero, accumName, valueName, body, typ) => ArrayFold(f(a), f(zero), accumName, valueName, f(body), typ)
    case MakeStruct(fields, _) => MakeStruct(fields map { case (n, a) => (n, f(a)) })
    case InsertFields(old, fields, _) => InsertFields(f(old), fields map { case (n, a) => (n, f(a)) })
    case GetField(o, name, typ) => GetField(f(o), name, typ)
    case GetFieldMissingness(o, name) => GetFieldMissingness(f(o), name)
    case AggIn(typ) => ir
    case AggMap(a, name, body, typ) => AggMap(f(a), name, f(body), typ)
    case AggFilter(a, name, body, typ) => AggFilter(f(a), name, f(body), typ)
    case AggFlatMap(a, name, body, typ) => AggFlatMap(f(a), name, f(body), typ)
    case ApplyAggOp(a, op, args, typ) => ApplyAggOp(f(a), op, args.map(f), typ)
    case MakeTuple(elts, typ) => MakeTuple(elts.map(f), typ)
    case GetTupleElement(tup, idx, typ) => GetTupleElement(f(tup), idx, typ)
    case In(i, typ) => ir
    case InMissingness(i) => ir
    case Die(message) => ir
    case Apply(fn, args, impl) => Apply(fn, args.map(f), impl)
    case ApplySpecial(fn, args, impl) => ApplySpecial(fn, args.map(f), impl)
    // from TableIR
    case TableCount(_) => ir
    case TableAggregate(child, query, typ) => TableAggregate(child, query, typ)
    case TableWrite(_, _, _, _) => ir
  }
}
