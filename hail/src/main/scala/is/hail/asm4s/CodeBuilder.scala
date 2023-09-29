package is.hail.asm4s

import is.hail.lir
import is.hail.asm4s._

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

  def scopedCode[T](mb: MethodBuilder[_])(f: (CodeBuilder) => Code[T]): Code[T] = {
    val (cbcode, retcode) = CodeBuilder.scoped(mb)(f)
    Code(cbcode, retcode)
  }

  def scopedVoid[T](mb: MethodBuilder[_])(f: (CodeBuilder) => Unit): Code[Unit] = {
    val (cbcode, _) = CodeBuilder.scoped(mb)(f)
    cbcode
  }
}

trait CodeBuilderLike {
  def mb: MethodBuilder[_]

  def isOpenEnded: Boolean

  // def code: Code[Unit] // debugging only

  def append(c: Code[Unit]): Unit

  def define(L: CodeLabel): Unit

  def result(): Code[Unit]

  def localBuilder: SettableBuilder = mb.localBuilder

  def fieldBuilder: SettableBuilder = mb.fieldBuilder

  def +=(c: Code[Unit]): Unit = append(c)
  def updateArray[T](array: Code[Array[T]], index: Code[Int], value: Code[T])(implicit tti: TypeInfo[T]): Unit = {
    append(array.update(index, value))
  }

  def memoize[T: TypeInfo](v: Code[T], optionalName: String = ""): Value[T] = v match {
    case b: ConstCodeBoolean => coerce[T](b.b)
    case _ => newLocal[T]("memoize" + optionalName, v)
  }

  def memoizeAny(v: Code[_], ti: TypeInfo[_]): Value[_] = {
    val l = newLocal("memoize")(ti)
    append(l.storeAny(v))
    l
  }

  def assign[T](s: Settable[T], v: Code[T]): Unit = {
    append(s := v)
  }

  def assignAny[T](s: Settable[T], v: Code[_]): Unit = {
    append(s := coerce[T](v))
  }

  def ifx(c: Code[Boolean], emitThen: => Unit): Unit =
    ifx(c, emitThen, Code._empty)

  def ifx(cond: Code[Boolean], emitThen: => Unit, emitElse: => Unit): Unit = {
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    val Lexit = CodeLabel()

    append(cond.branch(Ltrue, Lfalse))
    define(Ltrue)
    emitThen
    if (isOpenEnded) goto(Lexit)
    define(Lfalse)
    emitElse
    define(Lexit)
  }

  def loop(emitBody: CodeLabel => Unit): Unit = {
    val Lstart = CodeLabel()
    define(Lstart)
    emitBody(Lstart)
  }

  def whileLoop(cond: Code[Boolean], emitBody: CodeLabel => Unit): Unit =
    loop { Lstart =>
      emitBody(Lstart)
      ifx(cond, {
        emitBody(Lstart)
        goto(Lstart)
      })
    }

  def whileLoop(c: Code[Boolean], emitBody: => Unit): Unit =
    whileLoop(c, _ => emitBody)

  def forLoop(setup: => Unit, cond: Code[Boolean], incr: => Unit, emitBody: CodeLabel => Unit): Unit = {
    setup
    whileLoop(cond, {
      val Lincr = CodeLabel()
      emitBody(Lincr)
      define(Lincr)
      incr
    })
  }

  def forLoop(setup: => Unit, cond: Code[Boolean], incr: => Unit, body: => Unit): Unit =
    forLoop(setup, cond, incr, _ => body)

  def newLocal[T](name: String)(implicit tti: TypeInfo[T]): LocalRef[T] = mb.newLocal[T](name)

  def newLocal[T](name: String, c: Code[T])(implicit tti: TypeInfo[T]): LocalRef[T] = {
    val l = newLocal[T](name)
    append(l := c)
    l
  }

  def newLocalAny[T](name: String, c: Code[_])(implicit tti: TypeInfo[T]): LocalRef[T] =
    newLocal[T](name, coerce[T](c))

  def newField[T](name: String)(implicit tti: TypeInfo[T]): ThisFieldRef[T] = mb.genFieldThisRef[T](name)

  def newField[T](name: String, c: Code[T])(implicit tti: TypeInfo[T]): ThisFieldRef[T] = {
    val f = newField[T](name)
    append(f := c)
    f
  }

  def newFieldAny[T](name: String, c: Code[_])(implicit tti: TypeInfo[T]): ThisFieldRef[T] =
    newField[T](name, coerce[T](c))

  def goto(L: CodeLabel): Unit = {
    append(L.goto)
  }

  def _fatal(msgs: Code[String]*): Unit = {
    append(Code._fatal[Unit](msgs.reduce(_.concat(_))))
  }

  def _fatalWithError(errorId: Code[Int], msgs: Code[String]*): Unit = {
    append(Code._fatalWithID[Unit](msgs.reduce(_.concat(_)), errorId))
  }

  def _throw[T <: java.lang.Throwable](cerr: Code[T]): Unit = {
    append(Code._throw[T, Unit](cerr))
  }
}

class CodeBuilder(val mb: MethodBuilder[_], var code: Code[Unit]) extends CodeBuilderLike {
  def isOpenEnded: Boolean = {
    val last = code.end.last
    (last == null) || !last.isInstanceOf[lir.ControlX] || last.isInstanceOf[lir.ThrowX]
  }

  override def append(c: Code[Unit]): Unit = {
    assert(isOpenEnded)
    code = Code(code, c)
  }

  override def define(L: CodeLabel): Unit =
    if (isOpenEnded) append(L) else {
      val tmp = code
      code = new VCode(code.start, L.end, null)
      tmp.clear()
      L.clear()
    }

  def uncheckedAppend(c: Code[Unit]): Unit = {
    code = Code(code, c)
  }

  def result(): Code[Unit] = {
    val tmp = code
    code = Code._empty
    tmp
  }
}
