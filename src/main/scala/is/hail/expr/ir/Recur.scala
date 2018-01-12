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
    case MakeArrayN(len, elementType) => MakeArrayN(f(len), elementType)
    case ArrayRef(a, i, typ) => ArrayRef(f(a), f(i), typ)
    case ArrayMissingnessRef(a, i) => ArrayMissingnessRef(f(a), f(i))
    case ArrayLen(a) => ArrayLen(f(a))
    case ArrayMap(a, name, body, elementTyp) => ArrayMap(f(a), name, f(body), elementTyp)
    case ArrayFold(a, zero, accumName, valueName, body, typ) => ArrayFold(f(a), f(zero), accumName, valueName, f(body), typ)
    case MakeStruct(fields, _) => MakeStruct(fields map { case (n,a) => (n,f(a)) })
    case InsertFields(old, fields, _) => InsertFields(f(old), fields map { case (n,a) => (n,f(a)) } )
    case GetField(o, name, typ) => GetField(f(o), name, typ)
    case GetFieldMissingness(o, name) => GetFieldMissingness(f(o), name)
    case AggIn(typ) => ir
    case AggMap(a, name, body, typ) => AggMap(f(a), name, f(body), typ)
    case AggFilter(a, name, body, typ) => AggFilter(f(a), name, f(body), typ)
    case AggFlatMap(a, name, body, typ) => AggFlatMap(f(a), name, f(body), typ)
    case ApplyAggOp(a, op, args, typ) => ApplyAggOp(f(a), op, args.map(f), typ)
    case In(i, typ) => ir
    case InMissingness(i) => ir
    case Die(message) => ir
  }
}
