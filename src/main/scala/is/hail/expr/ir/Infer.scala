package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean, Type}
import is.hail.annotations.StagedRegionValueBuilder

object Infer {
  def apply(ir: IR) { apply(ir, Map()) }

  def apply(ir: IR, env: Map[String, Type]) {
    def infer(ir: IR, env: Map[String, Type] = env) { apply(ir, env) }
    def lookup(name: String): Type = {
      if (env.contains(name)) {
        env(name)
      } else {
        throw new RuntimeException(s"Cannot find key $name in env $env")
      }
    }
    ir match {
      case NA(t) =>
      case I32(x) =>
      case I64(x) =>
      case F32(x) =>
      case F64(x) =>
      case True() =>
      case False() =>

      case x@If(cond, cnsq, altr, _) =>
        infer(cond)
        infer(cnsq)
        infer(altr)
        assert(cond.typ == TBoolean)
        assert(cnsq.typ == altr.typ)
        x.typ = cnsq.typ
      case x@MapNA(name, value, body, _) =>
        infer(value)
        infer(body, env = env + (name -> value.typ))
        x.typ = body.typ
      case x@Let(name, value, body, _) =>
        infer(value)
        infer(body, env = env + (name -> value.typ))
        x.typ = body.typ
      case x@Ref(name, _) =>
        x.typ = lookup(name)
      case Set(name, v) =>
        infer(v)
        assert(lookup(name) == v.typ)
      case x@ApplyPrimitive(op, args, typ) =>
        args.map(infer(_))
        x.typ = Primitives.returnTyp(op, args.map(_.typ))
      case LazyApplyPrimitive(op, args, typ) =>
        ???
      case Lambda(name, paramTyp, body, typ) =>
        ???
      case x@MakeArray(args, _) =>
        args.map(infer(_))
        val t = args.head.typ
        args.map(_.typ).zipWithIndex.tail.foreach { case (x, i) => assert(x == t, s"at position $i type mismatch: $t $x") }
        x.typ = TArray(t)
      case MakeArrayN(len, _) =>
        infer(len)
        assert(len.typ.isInstanceOf[TArray])
      case x@ArrayRef(a, i, _) =>
        infer(a)
        infer(i)
        assert(i.typ == TInt32)
        x.typ = a.typ.asInstanceOf[TArray].elementType
      case ArrayLen(a) =>
        infer(a)
        assert(a.typ.isInstanceOf[TArray])
      case ArraySet(a, idx, v) =>
        infer(idx)
        assert(idx.typ == TInt32)
        infer(a)
        val t = a.typ.asInstanceOf[TArray].elementType
        infer(v)
        assert(v.typ == t)
      case x@For(value, idx, array, body) =>
        infer(array)
        val t = array.typ.asInstanceOf[TArray].elementType
        infer(body, env = env + (value -> t) + (idx -> TInt32))
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
      case x@Seq(stmts, _) =>
        stmts.foreach(infer(_))
        x.typ = stmts.last.typ
      case In(i, typ) =>
      case Out(v) =>
        infer(v)
    } }
}
