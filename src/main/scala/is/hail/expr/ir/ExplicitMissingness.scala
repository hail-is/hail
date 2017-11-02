package is.hail.expr.ir

import is.hail.utils.lensing._
import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean}
import is.hail.annotations.StagedRegionValueBuilder
import scala.collection.generic.Growable

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

object ExplicitMissingness {
  def apply(ir: IR): IR =
    apply(ir, new Env())

  def apply(ir: IR, env: Env[Unit]): IR =
    expression(ir, env)

  private def or(x: IR, y: IR): IR =
    ApplyPrimitive("||", Array(x, y), TBoolean)
  private def or(x: IR*): IR =
    x.foldLeft[IR](False())((x, y) => ApplyPrimitive("||", Array(x, y), TBoolean))
  private def present(x: IR): (IR, IR) =
    (False(), x)
  private def missing(x: IR): (IR, IR) =
    (True(), x)

  def expression(ir: IR, env: Env[Unit]): IR = {
    def expression(ir: IR, env: Env[Unit] = env): IR =
      ExplicitMissingness.expression(ir, env)
    ir match {
      case I32(x) =>
        ir
      case I64(x) =>
        ir
      case F32(x) =>
        ir
      case F64(x) =>
        ir
      case True() =>
        ir
      case False() =>
        ir

      case NA(typ) =>
        ir
      case MapNA(name, value, body, typ) =>
        MapNA(name, expression(value), expression(body, env = env.bind(name, ())), typ)
      case IsNA(value) =>
        IsNA(expression(value))

      case If(cond, cnsq, altr, typ) =>
        val name = env.freshName("cond")
        MapNA(name, expression(cond),
          If(Ref(name, TBoolean), expression(cnsq), expression(altr), typ), typ)
      case Let(name, value, body, typ) =>
        Let(name, expression(value), expression(body, env = env.bind(name, ())), typ)

      case Ref(name, typ) =>
        ir

      case ApplyPrimitive(op, args, typ) =>
        val typs = args map (_.typ)
        val names = env.freshNames((0 until args.length).map("x"+_):_*)
        val exp = ApplyPrimitive(op, (names zip typs) map Ref.tupled, typ)
        (names zip args).foldLeft[IR](exp) { case (exp, (name, v)) =>
          MapNA(name, expression(v), exp, typ)
        }
      case LazyApplyPrimitive(op, args, typ) =>
        // FIXME??? is it right that I don't check missingness?
        LazyApplyPrimitive(op, (args map (expression(_))), typ)
      case Lambda(names, body, null, null, typ) =>
        val (mnames, bodyenv) = env
          .bind(names map (_._1 -> (())):_*)
          .bindFresh(names map ("m" + _._1 -> (())):_*)
        Lambda(names, expression(body, env = bodyenv), mnames, null, typ)
      case Lambda(_, _, _, _, _) =>
        throw new UnsupportedOperationException(s"This lambda ($ir) already has missingness information?")
      case MakeArray(args, null, typ) =>
        MakeArray(args map (expression(_)), null, typ)
      case MakeArray(_, _, _) =>
        throw new UnsupportedOperationException(s"This array ($ir) already has missingness information?")
      case MakeArrayN(len, elementType) =>
        val name = env.freshName("cond")
        MapNA(name, expression(len),
          MakeArrayN(Ref(name, TBoolean), elementType), ir.typ)
      case ArrayRef(a, i, typ) =>
        val Array(xa, xi) = env.freshNames("a", "i")
        MapNA(xa, expression(a),
          MapNA(xi, expression(i),
            ArrayRef(Ref(xa, a.typ), Ref(xi, i.typ), typ), typ), typ)
      case ArrayMissingnessRef(a, i) =>
        val Array(xa, xi) = env.freshNames("a", "i")
        MapNA(xa, expression(a),
          MapNA(xi, expression(i),
            ArrayMissingnessRef(Ref(xa, a.typ), Ref(xi, i.typ)), TBoolean), TBoolean)
      case ArrayLen(a) =>
        val xa = env.freshName("a")
        MapNA(xa, expression(a),
          ArrayLen(Ref(xa, a.typ)), TInt32)
      case ArrayMap(a, lam, t) =>
        val xa = env.freshName("a")
        MapNA(xa, expression(a),
          ArrayMap(Ref(xa, a.typ), expression(lam), t), ir.typ)
      case ArrayFold(a, zero, lam, null, t) =>
        val xa = env.freshName("a")
        MapNA(xa, expression(a),
          ArrayFold(Ref(xa, a.typ), expression(zero), expression(lam), null, t), t)
      case ArrayFold(_, _, _, _, _) =>
        throw new UnsupportedOperationException(s"This fold ($ir) already has missingness information?")
      case MakeStruct(fields, null) =>
        MakeStruct(fields map on33(expression(_)))
      case MakeStruct(_, _) =>
        throw new UnsupportedOperationException(s"This struct ($ir) already has missingness information?")
      case GetField(o, name, typ) =>
        val xo = env.freshName("o")
        MapNA(xo, expression(o),
          GetField(Ref(xo, o.typ), name, typ), typ)
      case GetFieldMissingness(o, name) =>
        val xo = env.freshName("o")
        MapNA(xo, expression(o),
          GetFieldMissingness(Ref(xo, o.typ), name), TBoolean)
      case Seq(stmts, typ) =>
        Seq(stmts map (expression(_)), typ)
      case In(i, typ) =>
        ir
      case InMissingness(i) =>
        ir
      case Out(v) =>
        Out(expression(v))
      case Die(_) =>
        ir
    }
  }
}
