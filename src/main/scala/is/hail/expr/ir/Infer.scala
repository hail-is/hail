package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr.{TAggregable, TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean, Type, TVoid, TFunction, TNumeric, TSet}
import is.hail.annotations.StagedRegionValueBuilder

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
        x.typ = BinaryOp.inferReturnType(op, l.typ, r.typ)
      case x@ApplyUnaryPrimOp(op, v, _) =>
        infer(v)
        x.typ = UnaryOp.inferReturnType(op, v.typ)

      case x@MakeArray(args, _) =>
        args.map(infer(_))
        val t = args.head.typ
        args.map(_.typ).zipWithIndex.tail.foreach { case (x, i) => assert(x == t, s"at position $i type mismatch: $t $x") }
        x.typ = TArray(t)
      case MakeArrayN(len, _) =>
        infer(len)
        assert(len.typ.isOfType(TInt32()))
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
        infer(body, env.bind(accumName -> zero.typ, valueName -> tarray.elementType))
        assert(body.typ == zero.typ)
        x.typ = zero.typ

      case AggIn(typ) =>
        tAgg.foreach(x => assert(typ == x))
      case x@AggMap(a, name, body, _) =>
        infer(a)
        val tagg = a.typ.asInstanceOf[TAggregable]
        val env = Env.empty
          .bind(tagg.bindings:_*)
          .bind(name, tagg.elementType)
        infer(body, env = env)
        x.typ = tagg.copy(elementType = body.typ)
      case x@AggSum(a, _) =>
        infer(a)
        val tAgg = a.typ.asInstanceOf[TAggregable]
        x.typ = tAgg.elementType

      case MakeStruct(fields) =>
        fields.map { case (_, typ, v) =>
          infer(v)
          assert(typ == v.typ)
        }
      case x@GetField(o, name, _) =>
        infer(o)
        val t = o.typ.asInstanceOf[TStruct]
        assert(t.index(name).nonEmpty)
        x.typ = t.field(name).typ
      case GetFieldMissingness(o, name) =>
        infer(o)
        val t = o.typ.asInstanceOf[TStruct]
        assert(t.index(name).nonEmpty)

      case x@MakeSet(args, _) =>
        args.foreach(infer(_))
        if (args.isEmpty)
          assert(x.elementType != null)
        else {
          assert(args.tail.forall(_.typ == args(0).typ))
          x.elementType = args(0).typ
        }
      case x@SetAdd(set, element, elementType) =>
        infer(set)
        infer(element)
        val tSet = set.typ.asInstanceOf[TSet]
        assert(tSet.elementType == element.typ)
        x.elementType = tSet.elementType
      case SetContains(set, element) =>
        infer(set)
        infer(element)
        val tSet = set.typ.asInstanceOf[TSet]
        assert(tSet.elementType == element.typ)

      case In(i, typ) =>
        assert(typ != null)
      case InMissingness(i) =>
      case Die(msg) =>
    } }
}
