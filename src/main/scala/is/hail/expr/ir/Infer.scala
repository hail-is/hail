package is.hail.expr.ir

import is.hail.expr.types._

object Infer {
  def apply(ir: IR, tAgg: Option[TAggregable] = None) { apply(ir, tAgg, new Env[Type]()) }

  def apply(ir: IR, tAgg: Option[TAggregable], env: Env[Type]) {
    def infer(ir: IR, env: Env[Type] = env) { apply(ir, tAgg, env) }
    ir match {
      case I32(x) =>
      case I64(x) =>
      case F32(x) =>
      case F64(x) =>
      case True() =>
      case False() =>

      case Cast(v, typ) =>
        infer(v)
        assert(Casts.valid(v.typ, typ))

      case NA(t) =>
        assert(t != null)
      case x@MapNA(name, value, body, _) =>
        infer(value)
        infer(body, env = env.bind(name, value.typ))
        x.typ = body.typ
      case IsNA(v) =>
        infer(v)

      case x@If(cond, cnsq, altr, _) =>
        infer(cond)
        infer(cnsq)
        infer(altr)
        assert(cond.typ.isOfType(TBoolean()))
        assert(cnsq.typ == altr.typ, s"${cnsq.typ}, ${altr.typ}")
        x.typ = cnsq.typ

      case x@Let(name, value, body, _) =>
        infer(value)
        infer(body, env = env.bind(name, value.typ))
        println("let body: " + body.typ)
        x.typ = body.typ
      case x@Ref(_, _) =>
        x.typ = env.lookup(x)
      case x@ApplyBinaryPrimOp(op, l, r, _) =>
        infer(l)
        infer(r)
        x.typ = BinaryOp.getReturnType(op, l.typ, r.typ)
      case x@ApplyUnaryPrimOp(op, v, _) =>
        infer(v)
        x.typ = UnaryOp.getReturnType(op, v.typ)
      case x@MakeArray(args, typ) =>
        if (args.length == 0)
          assert(typ != null)
        else {
          args.foreach(infer(_))
          val t = args.head.typ
          args.map(_.typ).zipWithIndex.tail.foreach { case (x, i) => assert(x == t, s"at position $i type mismatch: $t $x") }
          x.typ = TArray(t)
        }
      case MakeArrayN(len, typ) =>
        infer(len)
        assert(len.typ.isOfType(TInt32()))
        assert(typ != null)
      case x@ArrayRef(a, i, _) =>
        infer(a)
        infer(i)
        assert(i.typ.isOfType(TInt32()))
        x.typ = coerce[TArray](a.typ).elementType
      case ArrayMissingnessRef(a, i) =>
        infer(a)
        infer(i)
        assert(i.typ.isOfType(TInt32()))
      case ArrayLen(a) =>
        infer(a)
        assert(a.typ.isInstanceOf[TArray])
      case x@ArrayMap(a, name, body, _) =>
        infer(a)
        val tarray = coerce[TArray](a.typ)
        infer(body, env = env.bind(name, tarray.elementType))
        x.elementTyp = body.typ
      case x@ArrayFold(a, zero, accumName, valueName, body, _) =>
        infer(a)
        val tarray = coerce[TArray](a.typ)
        infer(zero)
        infer(body, env = env.bind(accumName -> zero.typ, valueName -> tarray.elementType))
        assert(body.typ == zero.typ)
        x.typ = zero.typ
      case x@AggIn(typ) =>
        (tAgg, typ) match {
          case (Some(t), null) => x.typ = t
          case (Some(t), t2) => assert(t == t2)
          case (None, _) => throw new RuntimeException("must provide type of aggregable to Infer")
        }
      case x@AggMap(a, name, body, _) =>
        infer(a)
        val tagg = coerce[TAggregable](a.typ)
        infer(body, env = aggScope(tagg).bind(name, tagg.elementType))
        val tagg2 = tagg.copy(elementType = body.typ)
        tagg2.symTab = tagg.symTab
        x.typ = tagg2
      case x@AggFilter(a, name, body, typ) =>
        infer(a)
        val tagg = coerce[TAggregable](a.typ)
        infer(body, env = aggScope(tagg).bind(name, tagg.elementType))
        assert(body.typ.isInstanceOf[TBoolean])
        x.typ = tagg
      case x@AggFlatMap(a, name, body, typ) =>
        infer(a)
        val tagg = coerce[TAggregable](a.typ)
        infer(body, env = aggScope(tagg).bind(name, tagg.elementType))
        val tout = coerce[TArray](body.typ)
        val tagg2 = tagg.copy(elementType = tout.elementType)
        tagg2.symTab = tagg.symTab
        x.typ = tagg2
      case x@ApplyAggNullaryOp(a, op, _) =>
        infer(a)
        val tAgg = coerce[TAggregable](a.typ)
        x.typ = AggOp.getNullaryType(op, tAgg.elementType)
      case x@ApplyAggUnaryOp(a, op, arg1, _) =>
        infer(a)
        infer(arg1)
        val tAgg = coerce[TAggregable](a.typ)
        x.typ = AggOp.getUnaryType(op, arg1.typ, tAgg.elementType)
      case x@ApplyAggTernaryOp(a, op, arg1, arg2, arg3, _) =>
        infer(a)
        infer(arg1)
        infer(arg2)
        infer(arg3)
        val tAgg = coerce[TAggregable](a.typ)
        x.typ = AggOp.getTernaryType(op, arg1.typ, arg2.typ, arg3.typ, tAgg.elementType)
      case x@MakeStruct(fields, _) =>
        fields.foreach { case (name, a) => infer(a) }
        x.typ = TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*)
      case x@InsertFields(old, fields, _) =>
        infer(old)
        fields.foreach { case (name, a) => infer(a) }
        x.typ = fields.foldLeft(old.typ){ case (t, (name, a)) =>
          t match {
            case t2: TStruct =>
              t2.selfField(name) match {
                case Some(f2) => t2.updateKey(name, f2.index, a.typ)
                case None => t2.appendKey(name, a.typ)
              }
            case _ => TStruct(name -> a.typ)
          }
        }.asInstanceOf[TStruct]
      case x@GetField(o, name, _) =>
        infer(o)
        val t = coerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty)
        x.typ = t.field(name).typ
      case GetFieldMissingness(o, name) =>
        infer(o)
        val t = coerce[TStruct](o.typ)
        assert(t.index(name).nonEmpty)
      case In(i, typ) =>
        assert(typ != null)
      case InMissingness(i) =>
      case Die(msg) =>
    }
  }

  private def aggScope(t: TAggregable): Env[Type] =
    Env.empty.bind(t.bindings:_*)
}
