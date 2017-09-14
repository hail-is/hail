package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.asm4s.TypeInfo
import is.hail.asm4s.ucode.UCode
import is.hail.asm4s.ucode.ULocalRef
import is.hail.expr.Type
import is.hail.expr.TStruct
import is.hail.annotations._

import scala.language.existentials

object Compile {
  case class DetailedTypeInfo[T, UT : TypeInfo](injectNonMissing: Option[(TypeInfo[T], UCode => UCode)]) {
    val uti = typeInfo[UT]
  }

  def apply(x: IR, outTyps: Array[DetailedTypeInfo[_,_]]): UCode = {
    terminal(x, Map(), outTyps)
  }

  private[ir] def nonTerminal(x: IR, env: Map[String, (UCode, UCode)], outTyps: Array[DetailedTypeInfo[_,_]]): (UCode, UCode) = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps)
    def terminal(x: IR) = this.terminal(x, env, outTyps)
    x match {
      case NA(pti) =>
        ((ucode.True(), pti.point))
      case I32(x) =>
        (ucode.False(), ucode.I32(x))
      case I64(x) =>
        (ucode.False(), ucode.I64(x))
      case F32(x) =>
        (ucode.False(), ucode.F32(x))
      case F64(x) =>
        (ucode.False(), ucode.F64(x))
      case True() =>
        (ucode.False(), ucode.True())
      case False() =>
        (ucode.False(), ucode.False())
      case If(cond, cnsq, altr) =>
        val (mcond, vcond) = nonTerminal(cond)
        val (mcnsq, vcnsq) = nonTerminal(cnsq)
        val (maltr, valtr) = nonTerminal(altr)

        val missingness = ucode.Or(ucode.And(mcond, mcnsq), ucode.And(ucode.Not(mcond), maltr))
        val code = ucode.If(mcond,
          ucode.Erase(Code._throw(Code.newInstance[NullPointerException]())),
          ucode.If(vcond, vcnsq, valtr))

        (missingness, code)
      case Let(name, value, typ, body) =>
        val (mvalue, vvalue) = nonTerminal(value)

        val missingness = ucode.Let(mvalue, typeInfo[Boolean]) { xmvalue =>
          ucode.Let(vvalue, typ) { xvvalue =>
            this.nonTerminal(body, env + (name -> (xmvalue -> xvvalue)), outTyps)._1 } }
        val code = ucode.Let(mvalue, typeInfo[Boolean]) { xmvalue =>
          ucode.Let(vvalue, typ) { xvvalue =>
            this.nonTerminal(body, env + (name -> (xmvalue -> xvvalue)), outTyps)._2 } }

        (missingness, code)
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
      case MapNA(name, value, valueTyp, body, bodyTyp) =>
        val (mvalue, vvalue) = nonTerminal(value)

        val missingness = ucode.Let(vvalue, valueTyp) { xvvalue =>
          val (mbody, _) = this.nonTerminal(body, env + (name -> (ucode.False() -> xvvalue)), outTyps)
          ucode.Or(mvalue, mbody) }

        val code = ucode.Let(vvalue, valueTyp) { xvvalue =>
          val (_, vbody) = this.nonTerminal(body, env + (name -> (ucode.False() -> xvvalue)), outTyps)
          ucode.If(mvalue, bodyTyp.point, vbody) }

        (missingness, code)
      case In(i) =>
        ???
      case x =>
        throw new UnsupportedOperationException(s"$x is not a non-terminal IR")
    }
  }

  private[ir] def terminal(x: IR, env: Map[String, (UCode, UCode)], outTyps: Array[DetailedTypeInfo[_,_]]): UCode = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps)
    def terminal(x: IR) = this.terminal(x, env, outTyps)
    x match {
      case Out(values) =>
        assert(values.length == 1)

        val (mvalue, vvalue) = nonTerminal(values(0))

        outTyps(0).injectNonMissing match {
          case Some((ti, inject)) =>
            ucode.If(mvalue, ucode.Null(), inject(vvalue))

          case None =>
            ucode.If(mvalue,
              ucode.Erase(Code._throw(Code.newInstance[NullPointerException, String](
                s"tried to return a missing value via a primitive type: ${outTyps(0)}"))),
              vvalue)

        }
      case x =>
        throw new UnsupportedOperationException(s"$x is not a terminal IR")
    }
  }

}
