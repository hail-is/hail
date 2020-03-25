package is.hail.asm4s

abstract class SettableBuilder {
  def newSettable[T](name: String)(implicit tti: TypeInfo[T]): Settable[T]
}

object CodeBuilder {
  def apply(mb: MethodBuilder[_]): CodeBuilder = new CodeBuilder(mb, Code._empty)

  def apply(mb: MethodBuilder[_], code: Code[Unit]): CodeBuilder = new CodeBuilder(mb, code)

  def scoped[T](mb: MethodBuilder[_])(f: (CodeBuilder) => T): (Code[Unit], T) = {
    val cb = CodeBuilder(mb)
    val t = f(cb)
    (cb.result(), t)
  }
}

trait CodeBuilderLike {
  def mb: MethodBuilder[_]

  def append(c: Code[Unit]): Unit

  def result(): Code[Unit]

  val localBuilder: SettableBuilder = new SettableBuilder {
    def newSettable[T](name: String)(implicit tti: TypeInfo[T]): Settable[T] = mb.newLocal[T](name)
  }

  val fieldBuilder: SettableBuilder = new SettableBuilder {
    def newSettable[T](name: String)(implicit tti: TypeInfo[T]): Settable[T] = mb.genFieldThisRef[T](name)
  }

  def +=(c: Code[Unit]): Unit = append(c)

  def assign[T](s: Settable[T], v: Code[T]): Unit = {
    append(s := v)
  }

  def assignAny[T](s: Settable[T], v: Code[_]): Unit = {
    append(s := coerce[T](v))
  }

  def ifx(c: Code[Boolean], emitThen: => Unit): Unit = {
    val Ltrue = CodeLabel()
    val Lafter = CodeLabel()
    append(c.mux(Ltrue.goto, Lafter.goto))
    append(Ltrue)
    emitThen
    append(Lafter)
  }

  def ifx(c: Code[Boolean], emitThen: => Unit, emitElse: => Unit): Unit = {
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    val Lafter = CodeLabel()
    append(c.mux(Ltrue.goto, Lfalse.goto))
    append(Ltrue)
    emitThen
    append(Lafter.goto)
    append(Lfalse)
    emitElse
    append(Lafter)
  }

  def memoizeField[T](c: Code[T], name: String)(implicit tti: TypeInfo[T]): Settable[T] = {
    val f = mb.genFieldThisRef[T](name)
    append(f := c)
    f
  }

  def memoizeFieldAny[T](c: Code[_], name: String)(implicit tti: TypeInfo[T]): Settable[T] = memoizeField(coerce[T](c), name)

  def memoize[T](c: Code[T], name: String)(implicit tti: TypeInfo[T]): Settable[T] = {
    val l = mb.newLocal[T](name)
    append(l := c)
    l
  }

  def memoizeAny[T](c: Code[_], name: String)(implicit tti: TypeInfo[T]): Settable[T] = memoize(coerce[T](c), name)

  def goto(L: CodeLabel): Unit = {
    append(L.goto)
  }

  def define(L: CodeLabel): Unit = {
    append(L)
  }

  def _fatal(msg: Code[String]): Unit = {
    append(Code._fatal[Unit](msg))
  }
}

class CodeBuilder(val mb: MethodBuilder[_], var code: Code[Unit]) extends CodeBuilderLike {
  def append(c: Code[Unit]): Unit = {
    code = Code(code, c)
  }

  def result(): Code[Unit] = code
}
