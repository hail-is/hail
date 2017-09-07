package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.asm4s.TypeInfo
import is.hail.asm4s.ucode.UCodeM
import is.hail.asm4s.ucode.UCodeM._
import is.hail.asm4s.ucode.UCode
import is.hail.asm4s.ucode.ULocalRef
import is.hail.expr.Type
import is.hail.expr.TStruct
import is.hail.annotations._

object Compile {
  case class OptionUCode(missing: UCode, value: UCode)

  def apply(x: IR, outTyps: Array[MaybeGenericTypeInfo[_]]): UCodeM[Unit] =
    terminal(x, Map(), outTyps)

  private[ir] def nonTerminal(x: IR, env: Map[String, OptionUCode], outTyps: Array[MaybeGenericTypeInfo[_]]): OptionUCode = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps)
    def terminal(x: IR) = this.terminal(x, env, outTyps)
    x match {
      case Null() =>
        OptionUCode(ucode.True(), ucode.Null())
      case I32(x) =>
        OptionUCode(ucode.False(), ucode.I32(x))
      case I64(x) =>
        OptionUCode(ucode.False(), ucode.I64(x))
      case F32(x) =>
        OptionUCode(ucode.False(), ucode.F32(x))
      case F64(x) =>
        OptionUCode(ucode.False(), ucode.F64(x))
      case True() =>
        OptionUCode(ucode.False(), ucode.True())
      case False() =>
        OptionUCode(ucode.False(), ucode.False())
      case ApplyPrimitive(op, args) =>
        ???
      case LazyApplyPrimitive(op, args) =>
        ???
      case Lambda(name, body) =>
        ???
      case Ref(name) =>
        env(name)
      case MakeArray(args, typ) =>
        ???
        // ucode.NewInitializedArray(args map nonTerminal, typ)
      case ArrayRef(a, i, typ) =>
        ???
        // ucode.ArrayRef(nonTerminal(a), nonTerminal(i), typ)
      case MakeStruct(fields) =>
        ???
      case GetField(o, name) =>
        ???
      case In(i) =>
        ???
      case x =>
        throw new UnsupportedOperationException(s"$x is not a non-terminal IR")
    }
  }

  private[ir] def terminal(x: IR, env: Map[String, OptionUCode], outTyps: Array[MaybeGenericTypeInfo[_]]): UCodeM[Unit] = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps)
    def terminal(x: IR) = this.terminal(x, env, outTyps)
    x match {
      case If(cond, cnsq, altr) =>
        val OptionUCode(m, v) = nonTerminal(cond)
        UCodeM.mux(m, ucode.Null(),
          UCodeM.mux(v, terminal(cnsq), terminal(altr)))
      case Let(name, value, typ, body) =>
        val OptionUCode(m, v) = nonTerminal(value)
        for {
          mx <- newVar(m, typeInfo[Boolean])
          x <- newVar(v, typ)
          _ <- this.terminal(body, env + (name -> OptionUCode(mx.load(), x.load())), outTyps)
        } yield ()
      case MapNull(name, value, valueTyp, body) =>
        val OptionUCode(m, v) = nonTerminal(value)
        UCodeM.mux(m, ucode.Null(),
          newVar(v, valueTyp) flatMap (x => this.terminal(body, env + (name -> OptionUCode(ucode.False(), x.load())), outTyps)))
      case Out(values) =>
        assert(values.length == 1)
        val OptionUCode(m, v) = nonTerminal(values(0))
        UCodeM.mux(m, ucode.Null(), ucode.Erase(outTyps(0).castToGeneric(ucode.Reify(v))))
      case x =>
        throw new UnsupportedOperationException(s"$x is not a terminal IR")
    }
  }

}
