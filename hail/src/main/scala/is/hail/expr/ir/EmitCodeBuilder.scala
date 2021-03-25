package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir.functions.StringFunctions
import is.hail.lir
import is.hail.types.physical.{PCode, PSettable, PType, PValue}
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

  def scopedEmitCode(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => EmitCode): EmitCode = {
    val (cbcode, ec) = EmitCodeBuilder.scoped(mb)(f)
    EmitCode(cbcode, ec)
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
    assert(s.pt.equalModuloRequired(v.pt), s"type mismatch!\n  settable=${s.pt}\n     passed=${v.pt}")
    s.store(this, v)
  }

  def assign(s: EmitSettable, v: EmitCode): Unit = {
    s.store(this, v)
  }

  def assign(s: EmitSettable, v: IEmitCode): Unit = {
    s.store(this, v)
  }

  def assign(is: IndexedSeq[EmitSettable], ix: IndexedSeq[EmitCode]): Unit = {
    (is, ix).zipped.foreach { case (s, c) => s.store(this, c) }
  }

  def assign(s: PresentEmitSettable, v: PCode): Unit = {
    s.store(this, v)
  }

  def memoize(pc: PCode, name: String): PValue = pc.memoize(this, name)

  def memoizeField(pc: PCode, name: String): PValue = {
    val f = emb.newPField(name, pc.pt)
    assign(f, pc)
    f
  }

  def memoize(v: EmitCode, name: String): EmitValue = {
    val l = emb.newEmitLocal(name, v.pt)
    assign(l, v)
    l
  }

  def memoize(v: IEmitCode, name: String): EmitValue = {
    val l = emb.newEmitLocal(name, v.pt)
    assign(l, v)
    l
  }

  def memoizeField[T](ec: EmitCode, name: String): EmitValue = {
    val l = emb.newEmitField(name, ec.pt)
    l.store(this, ec)
    l
  }

  def memoizeField(v: IEmitCode, name: String): EmitValue = {
    val l = emb.newEmitField(name, v.pt)
    assign(l, v)
    l
  }

  private def _invoke[T](callee: EmitMethodBuilder[_], _args: Param*): Code[T] = {
    val expectedArgs = callee.emitParamTypes
    val args = _args.toArray
    if (expectedArgs.size != args.length)
      throw new RuntimeException(s"invoke ${ callee.mb.methodName }: wrong number of parameters: " +
        s"expected ${ expectedArgs.size }, found ${ args.length }")
    val codeArgs = args.indices.flatMap { i =>
      val arg = args(i)
      val pt = expectedArgs(i)
      (arg, pt) match {
        case (CodeParam(c), cpt: CodeParamType) =>
          if (c.ti != cpt.ti)
            throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
              s"\n  got ${ c.ti }" +
              s"\n  expected ${ cpt.ti }")
          FastIndexedSeq(c)
        case (PCodeParam(pc), pcpt: PCodeParamType) =>
          if (pc.pt != pcpt.pt)
            throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
              s"\n  got ${ pc.pt }" +
              s"\n  expected ${ pcpt.pt }")
          pc.codeTuple()
        case (EmitParam(ec), EmitParamType(pt)) =>
          if (!ec.pt.equalModuloRequired(pt)) {
            throw new RuntimeException(s"invoke ${callee.mb.methodName}: arg $i: type mismatch:" +
              s"\n  got ${ec.pt}" +
              s"\n  expected ${pt}")
          }

          val castEc = (ec.pt.required, pt.required) match {
            case (true, false) => {
              ec.map(pc => PCode(pc.pt.setRequired(pt.required), pc.code))
            }
            case (false, true) => {
              EmitCode.fromI(callee) { cb => IEmitCode.present(this, ec.toI(this).get(this))}
            }
            case _ => ec
          }

          if (castEc.pt.required) {
            append(castEc.setup)
            append(Code.toUnit(castEc.m))
            castEc.codeTuple()
          } else {
            val ev = memoize(castEc, "cb_invoke_setup_params")
            ev.codeTuple()
          }

        case (arg, expected) =>
          throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
            s"\n  got ${ arg }" +
            s"\n  expected ${ expected }")
      }
    }
    callee.mb.invoke(codeArgs: _*)
  }

  def invokeVoid(callee: EmitMethodBuilder[_], args: Param*): Unit = {
    assert(callee.emitReturnType == CodeParamType(UnitInfo))
    append(_invoke[Unit](callee, args: _*))
  }

  def invokeCode[T](callee: EmitMethodBuilder[_], args: Param*): Code[T] = {
    callee.emitReturnType match {
      case CodeParamType(UnitInfo) =>
        throw new AssertionError("CodeBuilder.invokeCode had unit return type, use invokeVoid")
      case _: CodeParamType =>
      case x => throw new AssertionError(s"CodeBuilder.invokeCode expects CodeParamType return, got $x")
    }
    _invoke[T](callee, args: _*)
  }

  def invokeEmit(callee: EmitMethodBuilder[_], args: Param*): IEmitCode = {
    val pt = callee.emitReturnType.asInstanceOf[EmitParamType].pt
    val r = newLocal("invokeEmit_r")(pt.codeReturnType())
    assignAny(r, _invoke(callee, args: _*))
    IEmitCode.fromCodeTuple(this, pt, Code.loadTuple(callee.modb, EmitCode.codeTupleTypes(pt), r))
  }

  // FIXME: this should be invokeSCode and should allocate/destructure a tuple when more than one code is present
  def invokePCode(callee: EmitMethodBuilder[_], args: Param*): PCode = {
    val pt = callee.emitReturnType.asInstanceOf[PCodeParamType].pt
    PCode(pt, _invoke(callee, args: _*))
  }

  // for debugging
  def printRegionValue(value: Code[_], typ: PType, region: Value[Region]): Unit = {
    append(Code._println(StringFunctions.boxArg(EmitRegion(emb, region), typ)(value)))
  }

  // for debugging
  def strValue(r: Value[Region], t: PType, code: Code[_]): Code[String] = {
    StringFunctions.boxArg(EmitRegion(emb, r), t)(code).invoke[String]("toString")
  }

  def strValue(r: Value[Region], x: PCode): Code[String] = strValue(r, x.pt, x.code)

  // for debugging
  def println(cString: Code[String]*) = this += Code._printlns(cString:_*)
}
