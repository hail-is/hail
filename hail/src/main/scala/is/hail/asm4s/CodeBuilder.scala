package is.hail.asm4s

import is.hail.expr.types.physical
import is.hail.lir

abstract class SettableBuilder {
  def newSettable[T](name: String)(implicit tti: TypeInfo[T]): Settable[T]
}

object CodeBuilder {
  def scoped[T](mb: MethodBuilder[_])(f: (CodeBuilder) => T): (Code[Unit], T) = {
    val start = new lir.Block
    val cb = new CodeBuilder(mb, start)
    val t = f(cb)

    (new VCode(start, cb.getEnd, null), t)
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

  protected def uncheckedAppend(c: Code[Unit]): Unit

  def define(L: CodeLabel): Unit

  def clear(): Unit

  def getEnd: lir.Block

  def setEnd(newEnd: lir.Block): Unit

  def append(c: Code[Unit]): Unit = {
    assert(isOpenEnded)
    uncheckedAppend(c)
  }

  def localBuilder: SettableBuilder = mb.localBuilder

  def fieldBuilder: SettableBuilder = mb.fieldBuilder

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
    define(Ltrue)
    emitThen
    define(Lafter)
  }

  def ifx(c: Code[Boolean], emitThen: => Unit, emitElse: => Unit): Unit = {
    val Ltrue = CodeLabel()
    val Lfalse = CodeLabel()
    val Lafter = CodeLabel()
    append(c.mux(Ltrue.goto, Lfalse.goto))
    define(Ltrue)
    emitThen
    if (isOpenEnded) goto(Lafter)
    define(Lfalse)
    emitElse
    define(Lafter)
  }

  def whileLoop(c: Code[Boolean], emitBody: => Unit): Unit = {
    val Lstart = CodeLabel()
    val Lbody = CodeLabel()
    val Lafter = CodeLabel()
    define(Lstart)
    append(c.mux(Lbody.goto, Lafter.goto))
    define(Lbody)
    emitBody
    goto(Lstart)
    define(Lafter)
  }

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

  def _fatal(msg: Code[String]): Unit = {
    append(Code._fatal[Unit](msg))
  }

  def _throw[T <: java.lang.Throwable](cerr: Code[T]): Unit = {
    append(Code._throw[T, Unit](cerr))
  }
}

class CodeBuilder(val mb: MethodBuilder[_], var end: lir.Block) extends CodeBuilderLike {
  def isOpenEnded: Boolean = {
    (end != null) && ((end.last == null) || !end.last.isInstanceOf[is.hail.lir.ControlX])
  }

  def getEnd: lir.Block = {
    val ret = end
    clear()
    ret
  }

  def setEnd(newEnd: lir.Block) {
    assert(!isOpenEnded)
    end = newEnd
  }

  def clear(): Unit = { end = null }

  def uncheckedAppend(c: Code[Unit]): Unit = {
    end.append(lir.goto(c.start))
    end = c.end
    c.clear()
  }

  override def define(L: CodeLabel): Unit = {
    if (isOpenEnded) append(L.goto)
    end = L.end
  }
}
