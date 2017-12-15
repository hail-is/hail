package is.hail.expr.ir

import is.hail.expr.{TAggregable, TArray, TBoolean, TInt32, TStruct, Type}

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
        x.typ = a.typ.asInstanceOf[TArray].elementType
      case ArrayMissingnessRef(a, i) =>
        infer(a)
        infer(i)
        assert(i.typ.isOfType(TInt32()))
      case ArrayLen(a) =>
        infer(a)
        assert(a.typ.isInstanceOf[TArray])
      case x@ArrayMap(a, name, body, _) =>
        infer(a)
        val tarray = a.typ.asInstanceOf[TArray]
        infer(body, env = env.bind(name, tarray.elementType))
        x.elementTyp = body.typ
      case x@ArrayFold(a, zero, accumName, valueName, body, _) =>
        infer(a)
        val tarray = a.typ.asInstanceOf[TArray]
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
        val tagg = a.typ.asInstanceOf[TAggregable]
        infer(body, env = aggScope(tagg).bind(name, tagg.elementType))
        val tagg2 = tagg.copy(elementType = body.typ)
        tagg2.symTab = tagg.symTab
        x.typ = tagg2
      case x@AggFilter(a, name, body, typ) =>
        infer(a)
        val tagg = a.typ.asInstanceOf[TAggregable]
        infer(body, env = aggScope(tagg).bind(name, tagg.elementType))
        assert(body.typ.isInstanceOf[TBoolean])
        x.typ = tagg
      case x@AggFlatMap(a, name, body, typ) =>
        infer(a)
        val tagg = a.typ.asInstanceOf[TAggregable]
        infer(body, env = aggScope(tagg).bind(name, tagg.elementType))
        val tout = body.typ.asInstanceOf[TArray]
        val tagg2 = tagg.copy(elementType = tout.elementType)
        tagg2.symTab = tagg.symTab
        x.typ = tagg2
      case x@AggSum(a, _) =>
        infer(a)
        val tAgg = a.typ.asInstanceOf[TAggregable]
        x.typ = tAgg.elementType
      case x@MakeStruct(fields, _) =>
        fields.foreach { case (name, a) => infer(a) }
        x.typ = TStruct(fields.map { case (name, a) =>
          (name, a.typ)
        }: _*)
      case x@GetField(o, name, _) =>
        infer(o)
        val t = o.typ.asInstanceOf[TStruct]
        assert(t.index(name).nonEmpty)
        x.typ = t.field(name).typ
      case GetFieldMissingness(o, name) =>
        infer(o)
        val t = o.typ.asInstanceOf[TStruct]
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
