package is.hail.expr.ir

object Recur {
<<<<<<< da22eb32264d1a628647697185ee222c20734739
  def apply(f: IR => IR)(ir: IR): IR = Copy(ir, Children(ir).map {
    case c: IR => f(c)
    case c => c
  }).asInstanceOf[IR]
=======
  def apply(f: IR => IR)(ir: IR): IR = ir match {
    case I32(x) => ir
    case I64(x) => ir
    case F32(x) => ir
    case F64(x) => ir
    case True() => ir
    case False() => ir
    case Void() => ir
    case Cast(v, typ) => Cast(f(v), typ)
    case NA(typ) => ir
    case IsNA(value) => IsNA(f(value))
    case If(cond, cnsq, altr) => If(f(cond), f(cnsq), f(altr))
    case Let(name, value, body) => Let(name, f(value), f(body))
    case Ref(name, typ) => ir
    case ApplyBinaryPrimOp(op, l, r) => ApplyBinaryPrimOp(op, f(l), f(r))
    case ApplyUnaryPrimOp(op, x) => ApplyUnaryPrimOp(op, f(x))
    case MakeArray(args, typ) => MakeArray(args map f, typ)
    case ArrayRef(a, i) => ArrayRef(f(a), f(i))
    case ArrayLen(a) => ArrayLen(f(a))
    case ArrayRange(start, stop, step) => ArrayRange(f(start), f(stop), f(step))
    case ArrayMap(a, name, body) => ArrayMap(f(a), name, f(body))
    case ArrayFilter(a, name, cond) => ArrayFilter(f(a), name, f(cond))
    case ArrayFlatMap(a, name, body) => ArrayFlatMap(f(a), name, f(body))
    case ArrayFold(a, zero, accumName, valueName, body) => ArrayFold(f(a), f(zero), accumName, valueName, f(body))
    case ArrayFor(a, valueName, body) => ArrayFor(f(a), valueName, f(body))
    case MakeStruct(fields) => MakeStruct(fields map { case (n, a) => (n, f(a)) })
    case InsertFields(old, fields) => InsertFields(f(old), fields map { case (n, a) => (n, f(a)) })
    case GetField(o, name) => GetField(f(o), name)
    case AggIn(typ) => ir
    case AggMap(a, name, body) => AggMap(f(a), name, f(body))
    case AggFilter(a, name, body) => AggFilter(f(a), name, f(body))
    case AggFlatMap(a, name, body) => AggFlatMap(f(a), name, f(body))
    case SeqOp(a, i, agg) => SeqOp(f(a), i, agg)
    case Begin(xs) => Begin(xs.map(f))
    case ApplyAggOp(a, op, args) => ApplyAggOp(f(a), op, args.map(f))
    case MakeTuple(elts) => MakeTuple(elts.map(f))
    case GetTupleElement(tup, idx) => GetTupleElement(f(tup), idx)
    case In(i, typ) => ir
    case Die(message) => ir
    case Apply(fn, args) => Apply(fn, args.map(f))
    case ApplySpecial(fn, args) => ApplySpecial(fn, args.map(f))
    // from MatrixIR
    case MatrixWrite(_, _, _, _) => ir
    // from TableIR
    case TableCount(_) => ir
    case TableAggregate(child, query) => TableAggregate(child, f(query))
    case TableWrite(_, _, _, _) => ir
    case TableExport(_, _, _, _, _) => ir
  }
>>>>>>> loop over columns in IR in MatrixMapRows
}
