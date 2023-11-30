package is.hail.asm4s

import is.hail.lir
import is.hail.utils.toRichIterable
import org.objectweb.asm.Opcodes.{INVOKESTATIC, INVOKEVIRTUAL}

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

  def this_ : Value[_] =
    mb.this_

  def isOpenEnded: Boolean

  // def code: Code[Unit] // debugging only

  def append(c: Code[Unit]): Unit

  def define(L: CodeLabel): Unit

  def result(): Code[Unit]

  def localBuilder: SettableBuilder = mb.localBuilder

  def fieldBuilder: SettableBuilder = mb.fieldBuilder

  def +=(c: Code[Unit]): Unit = append(c)

  def updateArray[T: TypeInfo](array: Code[Array[T]], index: Code[Int], value: Code[T]): Unit =
    append(array.update(index, value))

  def memoize[T: TypeInfo](v: Code[T], optionalName: String = "")
                          (implicit ev: T =!= Unit)
  : Value[T] =
    v match {
      case b: ConstCodeBoolean => coerce[T](b.b)
      case _ => newLocal[T]("memoize" + optionalName, v)
    }

  def memoizeAny(v: Code[_], ti: TypeInfo[_]): Value[_] =
    memoize(v.asInstanceOf[Code[AnyVal]])(ti.asInstanceOf[TypeInfo[AnyVal]], implicitly[AnyVal =!= Unit])

  def assign[T](s: Settable[T], v: Code[T]): Unit =
    append(s := v)

  def assignAny[T](s: Settable[T], v: Code[_]): Unit =
    append(s := coerce[T](v))

  /*
  Note [Evidence Is Unit]
  -----------------------
  Here's an example of a common `CodeBuilderLike` foot-gun:

    // previously:
    //   def ifx(cond: Code[Bool], csq: => Unit): Unit

    cb.ifx(cond, a := expr)

  What's wrong?
   -> It doesn't generate the right code despite passing the type-checker!
   -> `a := expr` is never emitted because `ifx` evaluates its parameters for effects only;
   -> their values are discarded by a conversion to `Unit`. [1]

  How do we fix this? We could write a test and catch this at runtime, but some errors will
  slip through. Better to use the type system to prevent these kind of foot-guns.

  The key observation is that the compiler inserts conversions to Unit. We can prevent it
  from doing this if we parameterise the type of `csq`:

  // def ifx[A](code: Code[Bool], csq: => A)(implicit ev: A =:= Unit): Unit

  The compiler now infers the type `A` from `csq`; no conversions to `Unit` are made.
  We can use an implicit constraint on `A` to fail compilation if `A` is inferred to
  anything other than `Unit`, thus catching and preventing this foot-gun!

  It's worth bearing in mind that while this handles simple cases, it won't prevent the
  dedicated hacker from working around it.

  [1]: https://github.com/scala/scala/blob/2.13.x/spec/06-expressions.md#value-discarding
  */

  def if_[A](c: => Code[Boolean], emitThen: => A)
            (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit =
    if_(c, emitThen, ().asInstanceOf[A])

  def if_[A](cond: => Code[Boolean], emitThen: => A, emitElse: => A)
            (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit = {
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

  def switch[A](discriminant: => Code[Int], emitDefault: => A, cases: IndexedSeq[() => A])
               (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit = {
    val Lexit = CodeLabel()
    val Lcases = IndexedSeq.fill(cases.length)(CodeLabel())
    val Ldefault = CodeLabel()

    append(discriminant.switch(Ldefault, Lcases))
    (Lcases, cases).zipped.foreach { case (label, emitCase) =>
      define(label)
      emitCase()
      if (isOpenEnded) append(Lexit.goto)
    }
    define(Ldefault)
    emitDefault
    define(Lexit)
  }

  def loop[A](emitBody: CodeLabel => A)
             (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit = {
    val Lstart = CodeLabel()
    define(Lstart)
    emitBody(Lstart)
  }

  def while_[A](cond: => Code[Boolean], emitBody: CodeLabel => A)
               (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit =
    loop { Lstart =>
      if_(cond, {
        emitBody(Lstart)
        goto(Lstart)
      })
    }

  def while_[A](c: => Code[Boolean], emitBody: => A)
               (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit =
    while_(c, (_: CodeLabel) => emitBody)

  def for_[A](setup: => A, cond: => Code[Boolean], incr: => A, emitBody: CodeLabel => A)
             (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit = {
    setup
    while_(cond, {
      val Lincr = CodeLabel()
      emitBody(Lincr)
      define(Lincr)
      incr
    })
  }

  def for_[A](setup: => A, cond: => Code[Boolean], incr: => A, body: => A)
             (implicit ev: A =:= Unit /* Note [Evidence Is Unit] */): Unit =
    for_(setup, cond, incr, (_: CodeLabel) => body)

  def newLocal[T: TypeInfo](name: String)(implicit ev: T =!= Unit)
  : LocalRef[T] =
    mb.newLocal[T](name)

  def newLocal[T: TypeInfo](name: String, c: Code[T])(implicit ev: T =!= Unit)
  : LocalRef[T] = {
    val l = newLocal[T](name)
    append(l := c)
    l
  }

  def newLocalAny(name: String, ti: TypeInfo[_]): LocalRef[_] =
    newLocal(name)(ti.asInstanceOf[TypeInfo[AnyVal]], implicitly[AnyVal =!= Unit])

  def newLocalAny(name: String, ti: TypeInfo[_], c: Code[_]): LocalRef[_] = {
    val ref = newLocalAny(name, ti)
    assignAny(ref, c)
    ref
  }

  def newField[T: TypeInfo](name: String)(implicit ev: T =!= Unit)
  : ThisFieldRef[T] =
    mb.genFieldThisRef[T](name)

  def newField[T: TypeInfo](name: String, c: Code[T])(implicit ev: T =!= Unit)
  : ThisFieldRef[T] = {
    val f = newField[T](name)
    append(f := c)
    f
  }

  def newFieldAny(name: String, ti: TypeInfo[_]): ThisFieldRef[_] =
    newField(name)(ti.asInstanceOf[TypeInfo[AnyVal]], implicitly[AnyVal =!= Unit])

  def newFieldAny(name: String, ti: TypeInfo[_], c: Code[_]): ThisFieldRef[_] = {
    val ref = newFieldAny(name, ti)
    assignAny(ref, c)
    ref
  }

  def goto(L: CodeLabel): Unit =
    append(L.goto)

  def _fatal(msgs: Code[String]*): Unit =
    append(Code._fatal[Unit](msgs.reduce(_.concat(_))))

  def _fatalWithError(errorId: Code[Int], msgs: Code[String]*): Unit =
    append(Code._fatalWithID[Unit](msgs.reduce(_.concat(_)), errorId))

  def _throw[T <: java.lang.Throwable](cerr: Code[T]): Unit =
    append(Code._throw[T, Unit](cerr))

  def _assert(cond: => Code[Boolean], message: Code[String]): Unit =
    if_(cond, {}, _throw(Code.newInstance[AssertionError, java.lang.Object](message)))
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

  def invoke[T](m: MethodBuilder[_], args: Value[_]*): Value[T] =
    if (m.returnTypeInfo eq UnitInfo) {
      append(invokeCode[Unit](m, args: _*))
      coerce[T](Code._empty)
    } else {
      val result = invokeCode[T](m, args: _*)
      memoize[T](result)(m.returnTypeInfo.asInstanceOf[TypeInfo[T]], implicitly[T =!= Unit])
    }

  def invokeCode[T](m: MethodBuilder[_], args: Value[_]*): Code[T] = {
    val (start, end, argvs) = Code.sequenceValues(args.toFastSeq.map(_.get))
    val op = if (m.isStatic) INVOKESTATIC else INVOKEVIRTUAL
    if (m.returnTypeInfo eq UnitInfo) {
      end.append(lir.methodStmt(op, m.lmethod, argvs))
      new VCode(start, end, null)
    } else {
      val value = lir.methodInsn(op, m.lmethod, argvs)
      new VCode(start, end, value)
    }
  }
}
