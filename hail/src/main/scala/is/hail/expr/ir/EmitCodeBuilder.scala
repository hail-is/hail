package is.hail.expr.ir

import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir.functions.StringFunctions
import is.hail.lir
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamValue}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.utils._

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

  def ifx[T: TypeInfo](c: Code[Boolean], emitThen: => Code[T], emitElse: => Code[T]): Value[T] = {
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    val Lafter = CodeLabel()
    append(c.mux(Ltrue.goto, Lfalse.goto))
    define(Ltrue)
    val tval = emitThen
    val value = newLocal[T]("ifx_value")
    assign(value, tval)
    goto(Lafter)
    define(Lfalse)
    assign(value, emitElse)
    define(Lafter)
    value
  }

  def ifx(c: Code[Boolean], emitThen: => SValue, emitElse: => SValue): SValue = {
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    val Lafter = CodeLabel()
    append(c.mux(Ltrue.goto, Lfalse.goto))
    define(Ltrue)
    val tval = emitThen
    val value = newSLocal(tval.st, "ifx_value")
    assign(value, tval)
    goto(Lafter)
    define(Lfalse)
    assign(value, emitElse)
    define(Lafter)
    value
  }

  def ifx(c: Code[Boolean], emitThen: => IEmitCode, emitElse: => IEmitCode): IEmitCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    append(c.mux(Ltrue.goto, Lfalse.goto))
    define(Ltrue)
    val tval = emitThen
    val value = newSLocal(tval.st, "ifx_value")
    tval.consume(this, {
      goto(Lmissing)
    }, { tval =>
      assign(value, tval)
      goto(Lpresent)
    })
    define(Lfalse)
    val fval = emitElse
    fval.consume(this, {
      goto(Lmissing)
    }, { fval =>
      assign(value, fval)
      goto(Lpresent)
    })
    IEmitCode(Lmissing, Lpresent, value, tval.required && fval.required)
  }

  def newSLocal(st: SType, name: String): SSettable = emb.newPLocal(name, st)

  def assign(s: SSettable, v: SValue): Unit = {
    assert(s.st == v.st, s"type mismatch!\n  settable=${s.st}\n     passed=${v.st}")
    s.store(this, v)
  }

  def assign(s: EmitSettable, v: EmitCode): Unit = {
    s.store(this, v)
  }

  def assign(s: EmitSettable, v: IEmitCode): Unit = {
    s.store(this, v)
  }

  def assign(is: IndexedSeq[EmitSettable], ix: IndexedSeq[EmitCode]): Unit = {
    (is, ix).zipped.foreach { (s, c) => s.store(this, c) }
  }

  def memoizeField(pc: SValue, name: String): SValue = {
    val f = emb.newPField(name, pc.st)
    assign(f, pc)
    f
  }

  def memoizeField[T: TypeInfo](v: Code[T], name: String): Value[T] = {
    newField[T](name, v)
  }

  def memoizeField[T: TypeInfo](v: Code[T]): Value[T] = {
    memoizeField[T](v, "memoize")
  }

  def memoizeFieldAny(v: Code[_], name: String, ti: TypeInfo[_]): Value[_] = {
    val l = newField(name)(ti)
    append(l.storeAny(v))
    l
  }

  def memoize(v: EmitCode): EmitValue =
    memoize(v, "memoize")

  def memoize(v: EmitCode, name: String): EmitValue = {
    require(v.st.isRealizable)
    val l = emb.newEmitLocal(name, v.emitType)
    assign(l, v)
    l
  }

  def memoize(v: IEmitCode): EmitValue =
    memoize(v, "memoize")

  def memoize(v: IEmitCode, name: String): EmitValue = {
    require(v.st.isRealizable)
    val l = emb.newEmitLocal(name, v.emitType)
    assign(l, v)
    l
  }

  def memoizeField[T](ec: EmitCode, name: String): EmitValue = {
    require(ec.st.isRealizable)
    val l = emb.newEmitField(name, ec.emitType)
    l.store(this, ec)
    l
  }

  def withScopedMaybeStreamValue[T](ec: EmitCode, name: String)(f: EmitValue => T): T = {
    if (ec.st.isRealizable) {
      f(memoizeField(ec, name))
    } else {
      assert(ec.st.isInstanceOf[SStream])
      val ev = if (ec.required)
        EmitValue(None, ec.toI(this).get(this, ""))
      else {
        val m = emb.genFieldThisRef[Boolean](name + "_missing")
        ec.toI(this).consume(this, assign(m, true), _ => assign(m, false))
        EmitValue(Some(m), ec.pv)
      }
      val res = f(ev)
      ec.pv match {
        case ss: SStreamValue => ss.defineUnusedLabels(emb)
      }
      res
    }
  }

  def memoizeField(v: IEmitCode, name: String): EmitValue = {
    require(v.st.isRealizable)
    val l = emb.newEmitField(name, v.emitType)
    assign(l, v)
    l
  }

  private def _invoke[T](callee: EmitMethodBuilder[_], _args: Param*): Value[T] = {

    // Instance methods must supply `this` in first position.
    val expectedArgs =
      if (callee.mb.isStatic) callee.emitParamTypes
      else CodeParamType(callee.ecb.cb.ti) +: callee.emitParamTypes

    val args = _args.toArray

    if (expectedArgs.size != args.length)
      throw new RuntimeException(s"invoke ${callee.mb.methodName}: wrong number of parameters: " +
        s"expected ${expectedArgs.size}, found ${args.length}"
      )

    val codeArgs = args.zip(expectedArgs).zipWithIndex.flatMap { case ((arg, pt), i) =>
      (arg, pt) match {
        case (CodeParam(c), cpt: CodeParamType) =>
          if (c.ti != cpt.ti)
            throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
              s"\n  got ${ c.ti }" +
              s"\n  expected ${ cpt.ti }" +
              s"\n  all param types: ${expectedArgs}-")
          FastSeq(c)
        case (SCodeParam(pc), pcpt: SCodeParamType) =>
          if (pc.st != pcpt.st)
            throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
              s"\n  got ${ pc.st }" +
              s"\n  expected ${ pcpt.st }")
          pc.valueTuple
        case (EmitParam(ec), SCodeEmitParamType(et)) =>
          if (!ec.emitType.equalModuloRequired(et)) {
            throw new RuntimeException(s"invoke ${callee.mb.methodName}: arg $i: type mismatch:" +
              s"\n  got ${ec.st}" +
              s"\n  expected ${et.st}")
          }

          val castEc = (ec.required, et.required) match {
            case (true, false) => ec.setOptional
            case (false, true) =>
              EmitCode.fromI(emb) { cb => IEmitCode.present(cb, ec.toI(cb).get(cb)) }
            case _ => ec
          }
          val castEv = memoize(castEc, "_invoke")
          castEv.valueTuple()
        case (arg, expected) =>
          throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
            s"\n  got ${ arg }" +
            s"\n  expected ${ expected }")
      }
    }

    callee.mb.invoke[T](this, codeArgs: _*)
  }

  def invokeVoid(callee: EmitMethodBuilder[_], args: Param*): Unit = {
    assert(callee.emitReturnType == CodeParamType(UnitInfo))
    append(_invoke[Unit](callee, args: _*))
  }

  def invokeCode[T](callee: EmitMethodBuilder[_], args: Param*): Value[T] = {
    callee.emitReturnType match {
      case CodeParamType(UnitInfo) =>
        throw new AssertionError("CodeBuilder.invokeCode had unit return type, use invokeVoid")
      case _: CodeParamType =>
      case x => throw new AssertionError(s"CodeBuilder.invokeCode expects CodeParamType return, got $x")
    }
    _invoke[T](callee, args: _*)
  }

  def invokeSCode(callee: EmitMethodBuilder[_], args: Param*): SValue = {
    val st = callee.emitReturnType.asInstanceOf[SCodeParamType].st
    if (st.nSettables == 1)
      st.fromValues(FastSeq(_invoke(callee, args: _*)))
    else {
      val tup = _invoke(callee, args: _*)
      st.fromValues(callee.asmTuple.loadElementsAny(tup))
    }
  }

  // for debugging
  def strValue(sc: SValue): Code[String] = {
    StringFunctions.svalueToJavaValue(this, emb.partitionRegion, sc).invoke[String]("toString")
  }

  def strValue(ec: EmitCode): Code[String] = {
    val s = newLocal[String]("s")
    ec.toI(this).consume(this, assign(s, "NA"), sc => assign(s, strValue(sc)))
    s
  }

  // for debugging
  def println(cString: Code[String]*) = this += Code._printlns(cString: _*)

  def logInfo(cs: Code[String]*): Unit = {
    this += Code.invokeScalaObject1[String, Unit](LogHelper.getClass, "logInfo", cs.reduce[Code[String]] { case (l, r) => (l.concat(r)) })
  }

  def warning(cs: Code[String]*): Unit = {
    this += Code.invokeScalaObject1[String, Unit](LogHelper.getClass, "warning", cs.reduce[Code[String]] { case (l, r) => (l.concat(r)) })
  }

  def consoleInfo(cs: Code[String]*): Unit = {
    this += Code.invokeScalaObject1[String, Unit](LogHelper.getClass, "consoleInfo", cs.reduce[Code[String]] { case (l, r) => (l.concat(r)) })
  }
}
