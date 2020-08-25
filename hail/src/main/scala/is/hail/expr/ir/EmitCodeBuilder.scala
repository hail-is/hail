package is.hail.expr.ir

import is.hail.asm4s.{coerce => _, _}
import is.hail.lir
import is.hail.types.physical.{PCode, PSettable, PValue}
import is.hail.utils.FastIndexedSeq

object EmitCodeBuilder {
  def apply(mb: EmitMethodBuilder[_]): EmitCodeBuilder = new EmitCodeBuilder(mb, Code._empty)

  def apply(mb: EmitMethodBuilder[_], code: Code[Unit]): EmitCodeBuilder = new EmitCodeBuilder(mb, code)

  def scoped[T](mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => T): (Code[Unit], T) = {
    val cb = EmitCodeBuilder(mb)
    val t = f(cb)
    (cb.result(), t)
  }

  def scopedCode[T](mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => Code[T]): Code[T] = {
    val (cbcode, retcode) = EmitCodeBuilder.scoped(mb)(f)
    Code(cbcode, retcode)
  }

  def scopedVoid(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => Unit): Code[Unit] = {
    val (cbcode, _) = EmitCodeBuilder.scoped(mb)(f)
    cbcode
  }
}

class EmitCodeBuilder(val emb: EmitMethodBuilder[_], var code: Code[Unit]) extends CodeBuilderLike {
  def isOpenEnded: Boolean = {
    val last = code.end.last
    (last == null) || !last.isInstanceOf[lir.ControlX] || last.isInstanceOf[lir.ThrowX]
  }

  def mb: MethodBuilder[_] = emb.mb

  def uncheckedAppend(c: Code[Unit]): Unit = {
    code = Code(code, c)
  }

  def result(): Code[Unit] = {
    val tmp = code
    code = Code._empty
    tmp
  }

  def assign(s: PSettable, v: PCode): Unit = {
    append(s := v)
  }

  def assign(s: EmitSettable, v: EmitCode): Unit = {
    append(s := v)
  }

  def assign(s: EmitSettable, v: IEmitCode): Unit = {
    s.store(this, v)
  }

  def memoize[T](pc: PCode, name: String): PValue = pc.memoize(this, name)

  def memoizeField[T](pc: PCode, name: String): PValue = {
    val f = emb.newPField(name, pc.pt)
    append(f := pc)
    f
  }

  def memoize[T](ec: EmitCode, name: String): EmitValue = {
    val l = emb.newEmitLocal(name, ec.pt)
    append(l := ec)
    l
  }

  def memoizeField[T](ec: EmitCode, name: String): EmitValue = {
    val l = emb.newEmitField(name, ec.pt)
    append(l := ec)
    l
  }

  private def _invoke[T](callee: EmitMethodBuilder[_], args: Param*): Code[T] = {
      val codeArgs = args.flatMap {
        case CodeParam(c) =>
          FastIndexedSeq(c)
        case EmitParam(ec) =>
          if (ec.pt.required) {
            append(ec.setup)
            append(Code.toUnit(ec.m))
            ec.codeTuple()
          } else {
            val ev = memoize(ec, "cb_invoke_setup_params")
            ev.codeTuple()
          }
      }
      callee.mb.invoke(codeArgs: _*)
  }

  def invokeVoid(callee: EmitMethodBuilder[_], args: Param*): Unit = {
    assert(callee.emitReturnType == CodeParamType(UnitInfo))
    append(_invoke[Unit](callee, args: _*))
  }

  def invokeCode[T](callee: EmitMethodBuilder[_], args: Param*): Code[T] = {
    assert(callee.emitReturnType.isInstanceOf[CodeParamType])
    _invoke[T](callee, args: _*)
  }

  def invokeEmit(callee: EmitMethodBuilder[_], args: Param*): EmitCode = {
    val pt = callee.emitReturnType.asInstanceOf[EmitParamType].pt
    val r = newLocal("invokeEmit_r")(pt.codeReturnType())
    EmitCode(r := _invoke(callee, args: _*),
      EmitCode.fromCodeTuple(pt, Code.loadTuple(callee.modb, EmitCode.codeTupleTypes(pt), r)))
  }
}
