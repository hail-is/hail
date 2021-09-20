package is.hail.expr.ir

import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.streams.StreamProducer
import is.hail.lir
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SValue}
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

  def assign(s: SSettable, v: SCode): Unit = {
    assert(s.st == v.st, s"type mismatch!\n  settable=${s.st}\n     passed=${v.st}")
    s.store(this, v)
  }

  def assign(s: SSettable, v: SValue): Unit = {
    assert(s.st == v.st, s"type mismatch!\n  settable=${s.st}\n     passed=${v.st}")
    s.store(this, v.get)
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

  def memoize(pc: SCode, name: String): SValue = pc.memoize(this, name)

  def memoizeField(pc: SCode, name: String): SValue = {
    val f = emb.newPField(name, pc.st)
    assign(f, pc)
    f
  }

  def memoize[T: TypeInfo](v: Code[T], optionalName: String = ""): Value[T] = {
    newLocal[T]("memoize" + optionalName, v)
  }

  def memoizeField[T: TypeInfo](v: Code[T]): Value[T] = {
    newField[T]("memoize", v)
  }

  def memoizeAny(v: Code[_], ti: TypeInfo[_]): Value[_] = {
    val l = newLocal("memoize")(ti)
    append(l.storeAny(v))
    l
  }

  def memoize(v: EmitCode, name: String): EmitValue = {
    require(v.st.isRealizable)
    val l = emb.newEmitLocal(name, v.emitType)
    assign(l, v)
    l
  }

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
        case SStreamValue(_, producer) => StreamProducer.defineUnusedLabels(producer, emb)
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
              s"\n  expected ${ cpt.ti }" +
              s"\n  all param types: ${expectedArgs}-")
          FastIndexedSeq(c)
        case (SCodeParam(pc), pcpt: SCodeParamType) =>
          if (pc.st != pcpt.st)
            throw new RuntimeException(s"invoke ${ callee.mb.methodName }: arg $i: type mismatch:" +
              s"\n  got ${ pc.st }" +
              s"\n  expected ${ pcpt.st }")
          memoize(pc, "_invoke").valueTuple.map(_.get)
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
          castEv.valueTuple().map(_.get)
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

  def invokeSCode(callee: EmitMethodBuilder[_], args: Param*): SCode = {
    val st = callee.emitReturnType.asInstanceOf[SCodeParamType].st
    if (st.nSettables == 1)
      st.fromValues(FastIndexedSeq(memoize(_invoke(callee, args: _*))(st.settableTupleTypes()(0)))).get
    else {
      val tup = newLocal("invokepcode_tuple", _invoke(callee, args: _*))(callee.asmTuple.ti)
      st.fromValues(callee.asmTuple.loadElementsAny(tup)).get
    }
  }

  // for debugging
  def strValue(sc: SValue): Code[String] = {
    StringFunctions.scodeToJavaValue(this, emb.partitionRegion, sc).invoke[String]("toString")
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

  def consoleInfo(cs: Code[String]*): Unit = {
    this += Code.invokeScalaObject1[String, Unit](LogHelper.getClass, "consoleInfo", cs.reduce[Code[String]] { case (l, r) => (l.concat(r)) })
  }
}
