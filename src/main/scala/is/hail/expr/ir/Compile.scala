package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.asm4s.TypeInfo
import is.hail.asm4s.ucode.UCodeM
import is.hail.asm4s.ucode.UCodeM._
import is.hail.asm4s.ucode.UCode
import is.hail.asm4s.ucode.ULocalRef
import is.hail.expr.Type

case class Compile() {
  def apply(x: IR): UCodeM[Unit] =
    terminal(x, Map())

  private[ir] def nonTerminal(x: IR, env: Map[String, ULocalRef]): UCode = x match {
    case I32(x) =>
      ucode.I32(x)
    case I64(x) =>
      ucode.I64(x)
    case F32(x) =>
      ucode.F32(x)
    case F64(x) =>
      ucode.F64(x)
    case True() =>
      ucode.True()
    case False() =>
      ucode.False()
    case ApplyPrimitive(op, args) =>
      ???
    case LazyApplyPrimitive(op, args) =>
      ???
    case Lambda(name, body) =>
      ???
    case Ref(name) =>
      env(name).load()
    case MakeArray(args, typ) =>
      ucode.NewInitializedArray(args map (nonTerminal(_, env)), typ)
    case ArrayRef(a, i, typ) =>
      ucode.ArrayRef(nonTerminal(a, env), nonTerminal(i, env), typ)
    case MakeStruct(fields) =>
      ???
    case GetField(o, name) =>
      ???
    case In(i) =>
      ???
    case x =>
      throw new UnsupportedOperationException(s"$x is not a non-terminal IR")
  }

  private[ir] def terminal(x: IR, env: Map[String, ULocalRef]): UCodeM[Unit] = x match {
    case _If(cond, cnsq, altr) =>
      UCodeM.mux(nonTerminal(cond, env), terminal(cnsq, env), terminal(altr, env))
    case Let(name, value, typ, body) =>
      newVar(nonTerminal(value, env), typ) flatMap (x => terminal(body, env + (name -> x)))
    case MapNull(name, value, body) =>
      ???
    case Out(values) =>
      assert(values.length == 1)
      nonTerminal(values(0), env)
    case x =>
      throw new UnsupportedOperationException(s"$x is not a terminal IR")
  }

}
