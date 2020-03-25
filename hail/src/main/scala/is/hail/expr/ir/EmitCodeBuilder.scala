package is.hail.expr.ir

import is.hail.asm4s.{Code, CodeBuilderLike, MethodBuilder, TypeInfo, Value}

object EmitCodeBuilder {
  def apply(mb: EmitMethodBuilder[_]): EmitCodeBuilder = new EmitCodeBuilder(mb, Code._empty)

  def apply(mb: EmitMethodBuilder[_], code: Code[Unit]): EmitCodeBuilder = new EmitCodeBuilder(mb, code)

  def scoped[T](mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => T): (Code[Unit], T) = {
    val cb = EmitCodeBuilder(mb)
    val t = f(cb)
    (cb.result(), t)
  }
}

class EmitCodeBuilder(emb: EmitMethodBuilder[_], var code: Code[Unit]) extends CodeBuilderLike {
  def mb: MethodBuilder[_] = emb.mb

  def append(c: Code[Unit]): Unit = {
    code = Code(code, c)
  }

  def result(): Code[Unit] = code

  def assign(s: PSettable, v: PCode): Unit = {
    append(s := v)
  }

  def assign(s: EmitSettable, v: EmitCode): Unit = {
    append(s := v)
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
}
