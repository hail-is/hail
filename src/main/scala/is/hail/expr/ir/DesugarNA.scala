package is.hail.expr.ir

import is.hail.utils.lensing._
import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean, TVoid}
import is.hail.annotations.StagedRegionValueBuilder
import scala.collection.generic.Growable

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

object DesugarNA {
  private def dummyValue(t: expr.Type): IR = t match {
    case TBoolean => False()
    case TInt32 => I32(0)
    case TInt64 => I64(0L)
    case TFloat32 => F32(0.0f)
    case TFloat64 => F64(0.0)
    case _ => I64(0L) // reference types
  }

  def apply(ir: IR): IR =
    apply(ir, new Env())

  def apply(ir: IR, env: Env[IR]): IR =
    expression(ir, env)._2

  private def or(x: IR, y: IR): IR =
    ApplyPrimitive("||", Array(x, y), TBoolean)
  private def and(x: IR, y: IR): IR =
    ApplyPrimitive("&&", Array(x, y), TBoolean)
  private def not(x: IR): IR =
    ApplyPrimitive("!", Array(x), TBoolean)
  private def or(x: IR*): IR =
    x.foldLeft[IR](False())((x, y) => ApplyPrimitive("||", Array(x, y), TBoolean))
  private def present(x: IR): (IR, IR) =
    (False(), x)
  private def missing(x: IR): (IR, IR) =
    (True(), x)

  def expression(ir: IR, env: Env[IR]): (IR, IR) = {
    def expression(ir: IR, env: Env[IR] = env): (IR, IR) =
      DesugarNA.expression(ir, env)
    def nonMissing(ir: IR, env: Env[IR] = env): IR =
      expression(ir, env)._2
    ir match {
      case I32(x) =>
        present(ir)
      case I64(x) =>
        present(ir)
      case F32(x) =>
        present(ir)
      case F64(x) =>
        present(ir)
      case True() =>
        present(ir)
      case False() =>
        present(ir)

      case NA(typ) =>
        missing(dummyValue(typ))
      case MapNA(name, value, body, typ) =>
        val (mvalue, vvalue) = expression(value)
        val (mbody, vbody) = expression(body, env = env.bind(name, False()))
        (or(mvalue, Let(name, vvalue, mbody, TBoolean)), If(mvalue, dummyValue(typ), Let(name, vvalue, vbody, typ), typ))
      case IsNA(value) =>
        present(expression(value)._1)

      case If(cond, cnsq, altr, typ) =>
        val vcond = nonMissing(cond)
        val (mcnsq, vcnsq) = expression(cnsq)
        val (maltr, valtr) = expression(altr)
        (or(and(vcond, mcnsq), and(not(vcond), maltr)), If(vcond, vcnsq, valtr, typ))
      case Let(name, value, body, typ) =>
        val (mvalue, vvalue) = expression(value)
        val (mbody, vbody) = expression(body, env = env.bind(name, mvalue))
        (mbody, Let(name, vvalue, vbody, typ))

      case x@Ref(name, typ) =>
        (env.lookup(x), x)

      case ApplyPrimitive(op, args, typ) =>
        present(ApplyPrimitive(op, args map (nonMissing(_)), typ))
      case LazyApplyPrimitive(op, args, typ) =>
        val (missing, newArgs) = (args map (expression(_))).unzip
        (or(missing:_*), LazyApplyPrimitive(op, newArgs, typ))
      case Lambda(_, _, null, _, _) =>
        throw new UnsupportedOperationException(s"Run ExplicitMissingness first $ir")
      case Lambda(names, body, mnames, null, typ) =>
        val bodyenv = env
          .bind(((names map (_._1)) zip mnames) map { case (x, y) => (x -> Ref(y, TBoolean)) }:_*)
          .bind(mnames map (_ -> False()):_*)
        val (mbody, vbody) = expression(body, env = bodyenv)
        present(Lambda(names, vbody, mnames, mbody, typ))
      case Lambda(_, _, _, _, _) =>
        throw new UnsupportedOperationException(s"Already ran DesguarNA $ir")
      case MakeArray(args, null, typ) =>
        val (missingness, newArgs) = (args map (expression(_))).unzip
        present(MakeArray(newArgs, missingness, typ))
      case MakeArray(_, _, _) =>
        throw new UnsupportedOperationException(s"This array ($ir) already has missingness information?")
      case MakeArrayN(len, elementType) =>
        present(MakeArrayN(nonMissing(len), elementType))
      case ArrayRef(a, i, typ) =>
        val va = nonMissing(a)
        val vi = nonMissing(i)
        (ArrayMissingnessRef(va, vi), ArrayRef(va, vi, typ))
      case ArrayMissingnessRef(a, i) =>
        present(ArrayMissingnessRef(nonMissing(a), nonMissing(i)))
      case ArrayLen(a) =>
        present(ArrayLen(nonMissing(a)))
      case ArrayMap(a, lam, typ) =>
        present(ArrayMap(nonMissing(a), nonMissing(lam), typ))
      case ArrayFold(a, zero, lam, null, typ) =>
        val va = nonMissing(a)
        val vlam@Lambda(names, _, mnames, mbody, typ) = nonMissing(lam)
        val (mzero, vzero) = expression(zero)
        // FIXME fold's missingness is complicated, right???
        (ArrayFold(va, vzero, Lambda(names, mbody, mnames, False(), TBoolean), mzero, TBoolean), ArrayFold(va, vzero, vlam, mzero, typ))
      case ArrayFold(_, _, _, _, _) =>
        throw new UnsupportedOperationException(s"This fold ($ir) already has missingness information?")

      case MakeStruct(fields, null) =>
        val (missingness, vfields) = (fields map { case (n, t, v) =>
          val (mv, vv) = expression(v)
          (mv, (n, t, vv))
        }).unzip
        present(MakeStruct(fields, missingness))
      case MakeStruct(_, _) =>
        throw new UnsupportedOperationException(s"This struct ($ir) already has missingness information?")
      case GetField(o, name, typ) =>
        val vo = nonMissing(o)
        (GetFieldMissingness(vo, name), GetField(vo, name, typ))
      case GetFieldMissingness(o, name) =>
        present(GetFieldMissingness(nonMissing(o), name))

      case Seq(stmts, typ) =>
        present(Seq(stmts map (nonMissing(_)), typ))
      case In(i, typ) =>
        (InMissingness(i), In(i, typ))
      case InMissingness(i) =>
        present(ir)
      case Out(v) =>
        val (mv, vv) = expression(v)
        present(If(mv, Die("cannot return a missing value"), Out(vv), TVoid))
      case Die(_) =>
        present(ir)
    }
  }
}
