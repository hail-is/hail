package is.hail.expr.ir

import is.hail.asm4s.{Code, CodeBuilder, CodeLabel, VCode}
import is.hail.expr.types.physical
import is.hail.expr.types.physical.{CRetCode, CUnitCode, PCode, PSettable, PThunk, PThunkCode, PValue}
import is.hail.lir

object EmitCodeBuilder {
  def scoped[T](mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => T): (Code[Unit], T) = {
    val start = new lir.Block
    val cb = new EmitCodeBuilder(mb, start)
    val t = f(cb)

    (new VCode(start, cb.getEnd, null), t)
  }

  def scopedCode[T](mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => Code[T]): Code[T] = {
    val (cbcode, retcode) = EmitCodeBuilder.scoped(mb)(f)
    Code(cbcode, retcode)
  }

  def scopedPCode(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => PCode): PCode = {
    val (cbcode, retcode) = EmitCodeBuilder.scoped(mb)(f)
    PCode(retcode.pt, Code(cbcode, retcode.code))
  }

  def scopedVoid(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => Unit): Code[Unit] = {
    val (cbcode, _) = EmitCodeBuilder.scoped(mb)(f)
    cbcode
  }
}

class EmitCodeBuilder(val emb: EmitMethodBuilder[_], _end: lir.Block) extends CodeBuilder(emb.mb, _end) {
  def assign(s: PSettable, v: PCode): Unit = {
    append(s := v)
  }

  def assign(s: EmitSettable, v: EmitCode): Unit = {
    append(s := v)
  }

  def ife[R <: physical.CCode](c: Code[Boolean], emitThen: => R, emitElse: => R): R = {
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    append(c.mux(Ltrue.goto, Lfalse.goto))
    define(Ltrue)
    val t = emitThen
    clear()
    define(Lfalse)
    val f = emitElse
    physical.CCode.merge(emb, t, f)
  }

  def ifr(c: Code[Boolean], emitThen: => CRetCode, emitElse: => CRetCode): PCode = {
    val ret = ife(c, emitThen, emitElse)
    define(ret.end)
    ret.value
  }

  def ret(value: PCode): CRetCode = CRetCode(this, value)

  def ret(): CUnitCode = CUnitCode(this)

  def force[R <: physical.CCode](thunk: PCode): R = {
    assert(isOpenEnded)
    end.append(lir.goto(thunk.asInstanceOf[PThunkCode].start))
    clear()
    thunk.asInstanceOf[PThunkCode].end.asInstanceOf[R]
  }

  def define(u: CUnitCode): Unit = {
    assert(!isOpenEnded)
    end = u.end
  }

  def extract(r: CRetCode): PCode = {
    define(r.end)
    r.value
  }

  def memoize[T](pc: PCode, name: String): PValue = pc.memoize(this, name)

  def memoizeField[T](pc: PCode, name: String): PValue = {
    val f = emb.newPField(name, pc.pt)
    append(f := pc)
    f
  }

  def memoize[T](ec: EmitCode, name: String): EmitValue = {
    val l = emb.newEmitLocal(name, ec.valueType)
    append(l := ec)
    l
  }

  def memoizeField[T](ec: EmitCode, name: String): EmitValue = {
    val l = emb.newEmitField(name, ec.valueType)
    append(l := ec)
    l
  }
}
