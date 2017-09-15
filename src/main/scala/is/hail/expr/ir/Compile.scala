package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.asm4s.TypeInfo
import is.hail.expr.Type
import is.hail.expr.TStruct
import is.hail.annotations._
import org.objectweb.asm.tree._
import org.objectweb.asm.Opcodes._

import scala.language.existentials

object Compile {
  case class DetailedTypeInfo[T, UT : TypeInfo](injectNonMissing: Option[(TypeInfo[T], Code[_] => Code[_])]) {
    val uti = typeInfo[UT]
  }

  class MissingBits(fb: FunctionBuilder[_]) {
    private var used = 0
    private var bits: LocalRef[Int] = null

    def newBit(x: Code[Boolean]): Code[Boolean] = {
      if (used >= 64 || bits == null) {
        bits = fb.newLocal[Int]
        fb.emit(bits.store(0))
        used = 0
      }

      fb.emit(bits.store(bits | (x.asInstanceOf[Code[Int]] << used)))

      val read = bits & (1 << used)
      used += 1
      read.asInstanceOf[Code[Boolean]]
    }
  }

  def apply(x: IR, outTyps: Array[DetailedTypeInfo[_,_]], fb: FunctionBuilder[_]) {
    terminal(x, Map(), outTyps, fb, new MissingBits(fb))
  }

  private[ir] def nonTerminal(x: IR, env: Map[String, (Code[Boolean], Code[_])], outTyps: Array[DetailedTypeInfo[_,_]], fb: FunctionBuilder[_], mb: MissingBits): (Code[Boolean], Code[_]) = {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps, fb, mb)
    def terminal(x: IR) = this.terminal(x, env, outTyps, fb, mb)
    x match {
      case NA(pti) =>
        (const(true), pti.point)
      case I32(x) =>
        (const(false), const(x))
      case I64(x) =>
        (const(false), const(x))
      case F32(x) =>
        (const(false), const(x))
      case F64(x) =>
        (const(false), const(x))
      case True() =>
        (const(false), const(true))
      case False() =>
        (const(false), const(false))
      case If(cond, cnsq, altr) =>
        val (mcond, vcond) = nonTerminal(cond)
        val (mcnsq, vcnsq) = nonTerminal(cnsq)
        val (maltr, valtr) = nonTerminal(altr)

        val missingness = mcnsq || maltr
        val code = mcond.mux(
          Code._throw(Code.newInstance[NullPointerException]()),
          vcond.asInstanceOf[Code[Boolean]].mux(vcnsq, valtr))

        (missingness, code)
      case Let(name, value, typ: TypeInfo[t], body) =>
        val (mvalue, vvalue) = nonTerminal(value)

        val x = mb.newBit(mvalue)
        val y = fb.newLocal()(typ)
        fb.emit(y.store(vvalue.asInstanceOf[Code[t]]))

        this.nonTerminal(body, env + (name -> (x -> y)), outTyps, fb, mb)
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
      case MapNA(name, value, valueTyp: TypeInfo[t], body, bodyTyp) =>
        // FIXME: maybe we can pass down the "null" label and jump directly there
        val (mvalue, vvalue) = nonTerminal(value)

        val lnonnull = new LabelNode
        val lafter = new LabelNode

        val x = mb.newBit(mvalue)
        val y = fb.newLocal()(valueTyp)
        fb.emit(y.store(vvalue.asInstanceOf[Code[t]]))

        val (mbody, vbody) =
          this.nonTerminal(body, env + (name -> (const(false) -> y.load())), outTyps, fb, mb)

        val missingness = x || mbody
        val code = x.mux(bodyTyp.point, vbody)

        (missingness, code)
      case In(i) =>
        ???
      case x =>
        throw new UnsupportedOperationException(s"$x is not a non-terminal IR")
    }
  }

  private[ir] def terminal(x: IR, env: Map[String, (Code[Boolean], Code[_])], outTyps: Array[DetailedTypeInfo[_,_]], fb: FunctionBuilder[_], mb: MissingBits) {
    def nonTerminal(x: IR) = this.nonTerminal(x, env, outTyps, fb, mb)
    def terminal(x: IR) = this.terminal(x, env, outTyps, fb, mb)
    x match {
      case Out(values) =>
        assert(values.length == 1)

        val (mvalue, vvalue) = nonTerminal(values(0))

        outTyps(0).injectNonMissing match {
          case Some((ti, inject)) =>
            fb.emit(mvalue.mux(Code._null, inject(vvalue)))

          case None =>
            fb.emit(mvalue.mux(
              Code._throw(Code.newInstance[NullPointerException, String](
                s"tried to return a missing value via a primitive type: ${outTyps(0)}")),
              vvalue))

        }
      case x =>
        throw new UnsupportedOperationException(s"$x is not a terminal IR")
    }
  }

}
