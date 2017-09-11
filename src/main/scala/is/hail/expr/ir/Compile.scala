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
  case class PP[L,R](l: L, r: R)
  case class DetailedTypeInfo[T, UT : TypeInfo](injectNonMissing: Option[(TypeInfo[T], UCode[_] => UCode[_])]) {
    val uti = typeInfo[UT]
  }

  def apply(x: IR, outTyps: Array[DetailedTypeInfo[_,_]]): UCode[_] = {
    terminal(x, Map(), outTyps)
  }

  private[ir] def nonTerminal(x: IR, env: Map[String, PP[UCode[_], UCode[_]]], outTyps: Array[DetailedTypeInfo[_,_]]): UCode[PP[UCode[_], UCode[_]]] = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps)
    def terminal(x: IR) = this.terminal(x, env, outTyps)
    x match {
      case NA(pti) =>
        ucode.Ret(PP(ucode.True(), pti.point))
      case I32(x) =>
        ucode.Ret(PP(ucode.False(), ucode.I32(x)))
      case I64(x) =>
        ucode.Ret(PP(ucode.False(), ucode.I64(x)))
      case F32(x) =>
        ucode.Ret(PP(ucode.False(), ucode.F32(x)))
      case F64(x) =>
        ucode.Ret(PP(ucode.False(), ucode.F64(x)))
      case True() =>
        ucode.Ret(PP(ucode.False(), ucode.True()))
      case False() =>
        ucode.Ret(PP(ucode.False(), ucode.False()))
      case If(cond, cnsq, altr) =>
        for {
          PP(mcond, vcond) <- nonTerminal(cond)
          PP(mcnsq, vcnsq) <- nonTerminal(cnsq)
          PP(maltr, valtr) <- nonTerminal(altr)
        } yield PP(ucode.Or(mcnsq, maltr),
          ucode.If(mcond,
            ucode.Erase(Code._throw(Code.newInstance[NullPointerException]())),
            ucode.If(vcond, vcnsq, valtr)))
      case Let(name, value, typ, body) =>
        for {
          PP(mvalue, vvalue) <- nonTerminal(value)
          (_, xmvalue) <- ucode.Var(mvalue, typeInfo[Boolean])
          (_, xvvalue) <- ucode.Var(vvalue, typ)
          PP(mbody, vbody) <- this.nonTerminal(body, env + (name -> PP(xmvalue, xvvalue)), outTyps)
        } yield PP(mbody, vbody)
      case ApplyPrimitive(op, args) =>
        ???
      case LazyApplyPrimitive(op, args) =>
        ???
      case Lambda(name, body) =>
        ???
      case Ref(name) =>
        ucode.Ret(env(name))
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
        for {
          PP(mvalue, vvalue) <- nonTerminal(value)
          PP(mbody, vbody) <- ucode.Let(vvalue, valueTyp) { (_, xvvalue) =>
            this.nonTerminal(body, env + (name -> PP(ucode.False(), xvvalue)), outTyps)
          }
        } yield PP(ucode.Or(mvalue, mbody), ucode.If(mvalue, bodyTyp.point, vbody))
      case In(i) =>
        ???
      case x =>
        throw new UnsupportedOperationException(s"$x is not a non-terminal IR")
    }
  }

  private[ir] def terminal(x: IR, env: Map[String, PP[UCode[_], UCode[_]]], outTyps: Array[DetailedTypeInfo[_,_]]): UCode[_] = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps)
    def terminal(x: IR) = this.terminal(x, env, outTyps)
    x match {
      case Out(values) =>
        assert(values.length == 1)

        for {
          PP(mvalue, vvalue) <- nonTerminal(values(0))
          _ <- outTyps(0).injectNonMissing match {
            case Some((ti, inject)) =>
              ucode.If(mvalue, ucode.Null(), inject(vvalue))

            case None =>
              ucode.If(mvalue,
                ucode.Erase(Code._throw(Code.newInstance[NullPointerException, String](
                  s"tried to return a missing value via a primitive type: ${outTyps(0)}"))),
                vvalue)

          }
        } yield ()
      case x =>
        throw new UnsupportedOperationException(s"$x is not a terminal IR")
    }
  }

}
