package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean, Type, TVoid, TFunction}
import is.hail.annotations.StagedRegionValueBuilder

object Infer {
  def apply(ir: IR) { apply(ir, new Env[Type]()) }

  def apply(ir: IR, env: Env[Type]) {
    def infer(ir: IR, env: Env[Type] = env) { apply(ir, env) }
    ir match {
      case I32(x) =>
      case I64(x) =>
      case F32(x) =>
      case F64(x) =>
      case True() =>
      case False() =>

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
        assert(cond.typ == TBoolean)
        assert(cnsq.typ == altr.typ, s"${cnsq.typ}, ${altr.typ}")
        x.typ = cnsq.typ

      case x@Let(name, value, body, _) =>
        infer(value)
        infer(body, env = env.bind(name, value.typ))
        println("let body: " + body.typ)
        x.typ = body.typ
      case x@Ref(_, _) =>
        x.typ = env.lookup(x)
      case x@ApplyPrimitive(op, args, typ) =>
        args.map(infer(_))
        x.typ = Primitives.returnTyp(op, args.map(_.typ))
      case x@MakeArray(args, _) =>
        args.map(infer(_))
        val t = args.head.typ
        args.map(_.typ).zipWithIndex.tail.foreach { case (x, i) => assert(x == t, s"at position $i type mismatch: $t $x") }
        x.typ = TArray(t)
      case MakeArrayN(len, _) =>
        infer(len)
        assert(len.typ == TInt32)
      case x@ArrayRef(a, i, _) =>
        infer(a)
        infer(i)
        assert(i.typ == TInt32)
        x.typ = a.typ.asInstanceOf[TArray].elementType
      case ArrayMissingnessRef(a, i) =>
        infer(a)
        infer(i)
        assert(i.typ == TInt32)
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
      case In(i, typ) =>
        assert(typ != null)
      case InMissingness(i) =>
      case Out(v) =>
        infer(v)
      case Die(msg) =>
    } }
}
