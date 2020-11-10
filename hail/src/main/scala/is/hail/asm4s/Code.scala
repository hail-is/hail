package is.hail.asm4s

import java.io.PrintStream
import java.lang.reflect

import is.hail.lir
import is.hail.utils._

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type

import scala.reflect.ClassTag

abstract class Thrower[T] {
  def apply[U](cerr: Code[T])(implicit uti: TypeInfo[U]): Code[U]
}

object LineNumber {
  val none: LineNumber = LineNumber(0)
}

case class LineNumber(v: Int) extends AnyVal

object Code {
  def void[T](v: lir.StmtX): Code[T] = {
    val L = new lir.Block()
    L.append(v)
    new VCode(L, L, null)
  }

  def void[T](c: Code[_], f: (lir.ValueX) => lir.StmtX): Code[T] = {
    c.end.append(f(c.v))
    val newC = new VCode(c.start, c.end, null)
    c.clear()
    newC
  }

  def void[T](c1: Code[_], c2: Code[_], f: (lir.ValueX, lir.ValueX) => lir.StmtX)(implicit line: LineNumber): Code[T] = {
    c2.end.append(f(c1.v, c2.v))
    c1.end.append(lir.goto(c2.start, line.v))
    val newC = new VCode(c1.start, c2.end, null)
    c1.clear()
    c2.clear()
    newC
  }

  def apply[T](v: lir.ValueX): Code[T] = {
    val L = new lir.Block()
    new VCode(L, L, v)
  }

  def apply[T](c: Code[_], f: (lir.ValueX) => lir.ValueX): Code[T] = {
    val newC = new VCode(c.start, c.end, f(c.v))
    c.clear()
    newC
  }

  def apply[T](c1: Code[_], c2: Code[_], f: (lir.ValueX, lir.ValueX) => lir.ValueX)(implicit line: LineNumber): Code[T] = {
    c1.end.append(lir.goto(c2.start, line.v))
    val newC = new VCode(c1.start, c2.end, f(c1.v, c2.v))
    c1.clear()
    c2.clear()
    newC
  }

  def apply[T](c1: Code[_], c2: Code[_], c3: Code[_], f: (lir.ValueX, lir.ValueX, lir.ValueX) => lir.ValueX)(implicit line: LineNumber): Code[T] = {
    c1.end.append(lir.goto(c2.start, line.v))
    c2.end.append(lir.goto(c3.start, line.v))
    val newC = new VCode(c1.start, c3.end, f(c1.v, c2.v, c3.v))
    c1.clear()
    c2.clear()
    c3.clear()
    newC
  }

  def sequenceValues(cs: IndexedSeq[Code[_]])(implicit line: LineNumber): (lir.Block, lir.Block, IndexedSeq[lir.ValueX]) = {
    val start = new lir.Block()
    val end = cs.foldLeft(start) { (end, c) =>
      end.append(lir.goto(c.start, line.v))
      c.end
    }
    val r = (start, end, cs.map(_.v))
    cs.foreach(_.clear())
    r
  }

  def sequence1[T](cs: IndexedSeq[Code[Unit]], v: Code[T])(implicit line: LineNumber): Code[T] = {
    val start = new lir.Block()
    val end = (cs :+ v).foldLeft(start) { (end, c) =>
      end.append(lir.goto(c.start, line.v))
      c.end
    }
    assert(end eq v.end)
    val newC = new VCode(start, end, v.v)
    cs.foreach(_.clear())
    v.clear()
    newC
  }

  def apply[T](c1: Code[Unit], c2: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1), c2)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2), c3)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[Unit], c4: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2, c3), c4)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[Unit], c4: Code[Unit], c5: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2, c3, c4), c5)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[Unit], c4: Code[Unit], c5: Code[Unit], c6: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2, c3, c4, c5), c6)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[Unit], c4: Code[Unit], c5: Code[Unit], c6: Code[Unit], c7: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2, c3, c4, c5, c6), c7)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[Unit], c4: Code[Unit], c5: Code[Unit], c6: Code[Unit], c7: Code[Unit], c8: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2, c3, c4, c5, c6, c7), c8)

  def apply[T](c1: Code[Unit], c2: Code[Unit], c3: Code[Unit], c4: Code[Unit], c5: Code[Unit], c6: Code[Unit], c7: Code[Unit], c8: Code[Unit], c9: Code[T])(implicit line: LineNumber): Code[T] =
    sequence1(FastIndexedSeq(c1, c2, c3, c4, c5, c6, c7, c8), c9)

  def apply(cs: Seq[Code[Unit]])(implicit line: LineNumber): Code[Unit] = {
    if (cs.isEmpty)
      Code(null: lir.ValueX)
    else {
      assert(cs.forall(_.v == null))
      val fcs = cs.toFastIndexedSeq
      sequence1(fcs.init, fcs.last)
    }
  }

  def foreach[A](it: Seq[A])(f: A => Code[Unit])(implicit line: LineNumber): Code[Unit] = Code(it.map(f))

  def newInstance[T <: AnyRef](parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit tct: ClassTag[T], line: LineNumber): Code[T] =
    newInstance(parameterTypes, args, 0)

  def newInstance[T <: AnyRef](parameterTypes: Array[Class[_]], args: Array[Code[_]], lineNumber: Int)(implicit tct: ClassTag[T], line: LineNumber): Code[T] = {
    val tti = classInfo[T]

    val tcls = tct.runtimeClass

    val c = tcls.getDeclaredConstructor(parameterTypes: _*)
    assert(c != null,
      s"no such method ${ tcls.getName }(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    val (start, end, argvs) = Code.sequenceValues(args)
    val linst = new lir.Local(null, "new_inst", tti)
    val newInstX = lir.newInstance(
      tti, Type.getInternalName(tcls), "<init>",
      Type.getConstructorDescriptor(c), tti, argvs, lineNumber)
    end.append(lir.store(linst, newInstX, lineNumber))

    new VCode(start, end, lir.load(linst, line.v))
  }

  def newInstance[C](cb: ClassBuilder[C], ctor: MethodBuilder[C], args: IndexedSeq[Code[_]])(implicit line: LineNumber): Code[C] = {
    val (start, end, argvs) = sequenceValues(args)

    val linst = new lir.Local(null, "new_inst", cb.ti)

    end.append(lir.store(linst, lir.newInstance(cb.ti, ctor.lmethod, argvs, line.v), line.v))

    new VCode(start, end, lir.load(linst, line.v))
  }

  def newInstance[T <: AnyRef]()(implicit tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber): Code[T] =
    newInstance[T](Array[Class[_]](), Array[Code[_]]())

  def newInstance[T <: AnyRef, A1](a1: Code[A1]
  )(implicit a1ct: ClassTag[A1], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def newInstance[T <: AnyRef, A1, A2](a1: Code[A1], a2: Code[A2]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def newInstance[T <: AnyRef, A1, A2, A3](a1: Code[A1], a2: Code[A2], a3: Code[A3]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3), line.v)

  def newInstance[T <: AnyRef, A1, A2, A3, A4](a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4))

  def newInstance[T <: AnyRef, A1, A2, A3, A4, A5](a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5))

  def newInstance7[T <: AnyRef, A1, A2, A3, A4, A5, A6, A7](a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6], a7: Code[A7]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], a7ct: ClassTag[A7], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass, a7ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5, a6, a7))

  def newInstance11[T <: AnyRef, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11](a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4],
    a5: Code[A5], a6: Code[A6], a7: Code[A7], a8: Code[A8], a9: Code[A9], a10: Code[A10], a11: Code[A11]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], a7ct: ClassTag[A7],
    a8ct: ClassTag[A8], a9ct: ClassTag[A9], a10ct: ClassTag[A10], a11ct: ClassTag[A11], tct: ClassTag[T], tti: TypeInfo[T], line: LineNumber
  ): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass, a7ct.runtimeClass,
      a8ct.runtimeClass, a9ct.runtimeClass, a10ct.runtimeClass, a11ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))

  def newArray[T](size: Code[Int])(implicit tti: TypeInfo[T], line: LineNumber): Code[Array[T]] =
    Code(size, lir.newArray(tti, line.v))

  def whileLoop(cond: Code[Boolean], body: Code[Unit]*)(implicit line: LineNumber): Code[Unit] = {
    val L = CodeLabel()
    Code(
      L,
      cond.mux(
        Code(
          Code(body.toFastIndexedSeq),
          L.goto),
        Code._empty))
  }

  def forLoop(init: Code[Unit], cond: Code[Boolean], increment: Code[Unit], body: Code[Unit])(implicit line: LineNumber): Code[Unit] = {
    Code(
      init,
      Code.whileLoop(cond,
        body,
        increment
      )
    )
  }

  def invokeScalaObject[S](cls: Class[_], method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]]
  )(implicit sct: ClassTag[S], line: LineNumber
  ): Code[S] = {
    val m = Invokeable.lookupMethod(cls, method, parameterTypes)(sct)
    val staticObj = FieldRef("MODULE$")(ClassTag(cls), ClassTag(cls), classInfo(ClassTag(cls)))
    m.invoke(staticObj.getField(), args)
  }

  def invokeScalaObject0[S](cls: Class[_], method: String
  )(implicit sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](), Array[Code[_]]())

  def invokeScalaObject1[A1, S](cls: Class[_], method: String, a1: Code[A1]
  )(implicit a1ct: ClassTag[A1], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def invokeScalaObject2[A1, A2, S](cls: Class[_], method: String, a1: Code[A1], a2: Code[A2]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array(a1, a2))

  def invokeScalaObject3[A1, A2, A3, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array(a1, a2, a3))

  def invokeScalaObject4[A1, A2, A3, A4, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array(a1, a2, a3, a4))

  def invokeScalaObject5[A1, A2, A3, A4, A5, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](
      cls, method, Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass), Array(a1, a2, a3, a4, a5))

  def invokeScalaObject6[A1, A2, A3, A4, A5, A6, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](
      cls, method, Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass), Array(a1, a2, a3, a4, a5, a6))

  def invokeScalaObject7[A1, A2, A3, A4, A5, A6, A7, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6], a7: Code[A7]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], a7ct: ClassTag[A7], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](
      cls, method, Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass, a7ct.runtimeClass), Array(a1, a2, a3, a4, a5, a6, a7))

  def invokeScalaObject8[A1, A2, A3, A4, A5, A6, A7, A8, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6], a7: Code[A7], a8: Code[A8]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], a7ct: ClassTag[A7], a8ct: ClassTag[A8], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](
      cls, method, Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass, a7ct.runtimeClass, a8ct.runtimeClass), Array(a1, a2, a3, a4, a5, a6, a7, a8))

  def invokeScalaObject13[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6], a7: Code[A7], a8: Code[A8],
    a9: Code[A9], a10: Code[A10], a11: Code[A11], a12: Code[A12], a13: Code[A13]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], a7ct: ClassTag[A7],
    a8ct: ClassTag[A8], a9ct: ClassTag[A9], a10ct: ClassTag[A10], a11ct: ClassTag[A11], a12ct: ClassTag[A12], a13ct: ClassTag[A13], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeScalaObject[S](
      cls, method,
      Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass, a7ct.runtimeClass, a8ct.runtimeClass,
        a9ct.runtimeClass, a10ct.runtimeClass, a11ct.runtimeClass, a12ct.runtimeClass, a13ct.runtimeClass),
      Array(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)
    )

  def invokeStatic[S](cls: Class[_], method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]]
  )(implicit sct: ClassTag[S], line: LineNumber
  ): Code[S] = {
    val m = Invokeable.lookupMethod(cls, method, parameterTypes)(sct)
    assert(m.isStatic)
    m.invoke(null, args)
  }

  def invokeStatic0[T, S](method: String
  )(implicit tct: ClassTag[T], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](), Array[Code[_]]())

  def invokeStatic1[T, A1, S](method: String, a1: Code[A1]
  )(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], line: LineNumber
  ): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def invokeStatic2[T, A1, A2, S](method: String, a1: Code[A1], a2: Code[A2]
  )(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], line: LineNumber
  ): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def invokeStatic3[T, A1, A2, A3, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3]
  )(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], line: LineNumber
  ): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3))

  def invokeStatic4[T, A1, A2, A3, A4, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4]
  )(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], line: LineNumber
  ): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4))

  def invokeStatic5[T, A1, A2, A3, A4, A5, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5]
  )(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], line: LineNumber
  ): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5))

  def _null[T >: Null](implicit tti: TypeInfo[T], line: LineNumber): Code[T] = Code(lir.insn0(ACONST_NULL, tti, line.v))

  def _empty: Code[Unit] = Code[Unit](null: lir.ValueX)

  def _throwAny[T <: java.lang.Throwable]: Thrower[T] = new Thrower[T] {
    def apply[U](cerr: Code[T])(implicit uti: TypeInfo[U], line: LineNumber): Code[U] = {
      if (uti eq UnitInfo) {
        cerr.end.append(lir.throwx(cerr.v))
        val newC = new VCode(cerr.start, cerr.end, null)
        cerr.clear()
        newC
      } else
        Code(cerr, lir.insn1(ATHROW, uti, line.v))
    }
  }

  private def getEmitLineNum: LineNumber = {
    val st = Thread.currentThread().getStackTrace
    val i = st.indexWhere(ste => ste.getFileName == "Emit.scala")
    LineNumber(if (i == -1) 0 else st(i).getLineNumber)
  }

  def _throw[T <: java.lang.Throwable, U](cerr: Code[T])(implicit uti: TypeInfo[U], line: LineNumber): Code[U] = {
    // FIXME: should find a better way to decide if line numbers refer to Emit.scala or printed IR
    implicit val l = if (line.v == 0) getEmitLineNum else line
    if (uti eq UnitInfo) {
      cerr.end.append(lir.throwx(cerr.v, l.v))
      val newC = new VCode(cerr.start, cerr.end, null)
      cerr.clear()
      newC
    } else
      Code(cerr, lir.insn1(ATHROW, uti, l.v))
  }

  def _fatal[U](msg: Code[String])(implicit uti: TypeInfo[U], line: LineNumber): Code[U] = {
    val l = line
    val r = {
      // FIXME: should find a better way to decide if line numbers refer to Emit.scala or printed IR
      implicit val line = if (l.v == 0) getEmitLineNum else l
      val cerr = Code.newInstance[is.hail.utils.HailException, String, Option[String], Throwable](
        msg,
        Code.invokeStatic0[scala.Option[String], scala.Option[String]]("empty"),
        Code._null[Throwable])
      Code._throw[is.hail.utils.HailException, U](cerr)
    }
    r
  }

  def _fatalWithID[U](msg: Code[String], errorId: Int)(implicit uti: TypeInfo[U], line: LineNumber): Code[U] =
    Code._throw[is.hail.utils.HailException, U](Code.newInstance[is.hail.utils.HailException, String, Int](
      msg,
      errorId))

  def _return[T](c: Code[T])(implicit line: LineNumber): Code[Unit] = {
    c.end.append(if (c.v != null)
      lir.returnx(c.v, line.v)
    else
      lir.returnx(line.v))
    val newC = new VCode(c.start, c.end, null)
    c.clear()
    newC
  }

  def _printlns(cs: Code[String]*)(implicit line: LineNumber): Code[Unit] = {
    _println(cs.reduce[Code[String]] { case (l, r) => (l.concat(r)) })
  }

  def _println(c: Code[AnyRef])(implicit line: LineNumber): Code[Unit] = {
    Code(
      Code.invokeScalaObject1[AnyRef, Unit](scala.Console.getClass, "println", c),
      Code.invokeScalaObject0[Unit](scala.Console.getClass, "flush")
    )
  }

  def _assert(c: Code[Boolean])(implicit line: LineNumber): Code[Unit] =
    c.mux(Code._empty, Code._throw[AssertionError, Unit](Code.newInstance[AssertionError]()))

  def _assert(c: Code[Boolean], message: Code[String])(implicit line: LineNumber): Code[Unit] =
    c.mux(Code._empty, Code._throw[AssertionError, Unit](Code.newInstance[AssertionError, java.lang.Object](message)))

  def checkcast[T](v: Code[_])(implicit tti: TypeInfo[T], line: LineNumber): Code[T] =
    Code(v, lir.checkcast(tti.iname, line.v))

  def boxBoolean(cb: Code[Boolean])(implicit line: LineNumber): Code[java.lang.Boolean] = Code.newInstance[java.lang.Boolean, Boolean](cb)

  def boxInt(ci: Code[Int])(implicit line: LineNumber): Code[java.lang.Integer] = Code.newInstance[java.lang.Integer, Int](ci)

  def boxLong(cl: Code[Long])(implicit line: LineNumber): Code[java.lang.Long] = Code.newInstance[java.lang.Long, Long](cl)

  def boxFloat(cf: Code[Float])(implicit line: LineNumber): Code[java.lang.Float] = Code.newInstance[java.lang.Float, Float](cf)

  def boxDouble(cd: Code[Double])(implicit line: LineNumber): Code[java.lang.Double] = Code.newInstance[java.lang.Double, Double](cd)

  def booleanValue(x: Code[java.lang.Boolean])(implicit line: LineNumber): Code[Boolean] = toCodeObject(x).invoke[Boolean]("booleanValue")

  def intValue(x: Code[java.lang.Number])(implicit line: LineNumber): Code[Int] = toCodeObject(x).invoke[Int]("intValue")

  def longValue(x: Code[java.lang.Number])(implicit line: LineNumber): Code[Long] = toCodeObject(x).invoke[Long]("longValue")

  def floatValue(x: Code[java.lang.Number])(implicit line: LineNumber): Code[Float] = toCodeObject(x).invoke[Float]("floatValue")

  def doubleValue(x: Code[java.lang.Number])(implicit line: LineNumber): Code[Double] = toCodeObject(x).invoke[Double]("doubleValue")

  def getStatic[T: ClassTag, S: ClassTag : TypeInfo](field: String)(implicit line: LineNumber): Code[S] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.getField(null)
  }

  def putStatic[T: ClassTag, S: ClassTag : TypeInfo](field: String, rhs: Code[S])(implicit line: LineNumber): Code[Unit] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.put(null, rhs)
  }

  def currentTimeMillis()(implicit line: LineNumber): Code[Long] =
    Code.invokeStatic0[java.lang.System, Long]("currentTimeMillis")

  def memoize[T, U](c: Code[T], name: String)(f: (Value[T]) => Code[U]
  )(implicit tti: TypeInfo[T], line: LineNumber
  ): Code[U] = {
    if (c.start.first == null &&
      c.v != null) {
      c.v match {
        case v: lir.LdcX =>
          val t = new Value[T] {
            def get: Code[T] = Code(lir.ldcInsn(v.a, v.ti, line.v))
          }
          return f(t)
        // You can't forward local references here because the local might have changed
        // when the value is referenced in f.
        case _ =>
      }
    }

    val lr = new LocalRef[T](new lir.Local(null, name, tti))
    Code(lr := c, f(lr))
  }

  def memoizeAny[T, U](c: Code[_], name: String
  )(f: (Value[_]) => Code[U]
  )(implicit tti: TypeInfo[T], line: LineNumber
  ): Code[U] =
    memoize[T, U](coerce[T](c), name)(f)

  def memoize[T1, T2, U](c1: Code[T1], name1: String, c2: Code[T2], name2: String
  )(f: (Value[T1], Value[T2]) => Code[U]
  )(implicit t1ti: TypeInfo[T1], t2ti: TypeInfo[T2], line: LineNumber
  ): Code[U] =
    memoize(c1, name1)(v1 => memoize(c2, name2)(v2 => f(v1, v2)))

  def toUnit(c: Code[_]): Code[Unit] = {
    val newC = new VCode(c.start, c.end, null)
    c.clear()
    newC
  }

  def switch(c: Code[Int], dflt: Code[Unit], cases: IndexedSeq[Code[Unit]])(implicit line: LineNumber): Code[Unit] = {
    val L = new lir.Block()
    c.end.append(lir.switch(c.v, dflt.start, cases.map(_.start), line.v))
    dflt.end.append(lir.goto(L, line.v))
    cases.foreach(_.end.append(lir.goto(L, line.v)))
    val newC = new VCode(c.start, L, null)
    c.clear()
    dflt.clear()
    cases.foreach(_.clear())
    newC
  }

  def newLocal[T](name: String)(implicit tti: TypeInfo[T]): Settable[T] =
    new LocalRef[T](new lir.Local(null, name, tti))

  def newTuple(mb: MethodBuilder[_], elems: IndexedSeq[Code[_]])(implicit line: LineNumber): Code[_] = {
    val t = mb.modb.tupleClass(elems.map(_.ti))
    t.newTuple(elems)
  }

  def loadTuple(modb: ModuleBuilder, elemTypes: IndexedSeq[TypeInfo[_]], v: Value[_])(implicit line: LineNumber): IndexedSeq[Code[_]] = {
    val t = modb.tupleClass(elemTypes)
    t.loadElementsAny(v)
  }
}

trait Code[+T] {
  // val stack = Thread.currentThread().getStackTrace

  def start: lir.Block

  def end: lir.Block

  def v: lir.ValueX

  def check(): Unit

  def clear(): Unit

  def ti: TypeInfo[_] = {
    if (v == null)
      UnitInfo
    else
      v.ti
  }
}

class VCode[+T](
  var _start: lir.Block,
  var _end: lir.Block,
  var _v: lir.ValueX) extends Code[T] {
  // for debugging
  // val stack = Thread.currentThread().getStackTrace
  // var clearStack: Array[StackTraceElement] = _

  def start: lir.Block = {
    check()
    _start
  }

  def end: lir.Block = {
    check()
    _end
  }

  def v: lir.ValueX = {
    check()
    _v
  }

  def check(): Unit = {
    /*
    if (_start == null) {
      println(clearStack.mkString("\n"))
      println("-----")
      println(stack.mkString("\n"))
    }
     */
    assert(_start != null)
  }

  def clear(): Unit = {
    /*
    if (clearStack != null) {
      println(clearStack.mkString("\n"))
    }
    assert(clearStack == null)
    clearStack = Thread.currentThread().getStackTrace
     */

    _start = null
    _end = null
    _v = null
  }
}

object CodeKind extends Enumeration {
  type Kind = Value
  val V, C = Value
}

class CCode(
  private var _entry: lir.Block,
  private var _Ltrue: lir.Block,
  private var _Lfalse: lir.Block
)(implicit val line: LineNumber) extends Code[Boolean] {

  private var _kind: CodeKind.Kind = _

  private var _start: lir.Block = _
  private var _end: lir.Block = _
  private var _v: lir.ValueX = _

  def entry: lir.Block = {
    checkC()
    _entry
  }

  def Ltrue: lir.Block = {
    checkC()
    _Ltrue
  }

  def Lfalse: lir.Block = {
    checkC()
    _Lfalse
  }

  def start: lir.Block = {
    checkV()
    _start
  }

  def end: lir.Block = {
    checkV()
    _end
  }

  def v: lir.ValueX = {
    checkV()
    _v
  }

  private def checkV(): Unit = {
    if (_kind == null) {
      assert(_entry != null)
      val c = new lir.Local(null, "bool", BooleanInfo)
      _start = _entry
      _end = new lir.Block()
      _Ltrue.append(lir.store(c, lir.ldcInsn(1, BooleanInfo, line.v), line.v))
      _Ltrue.append(lir.goto(_end, line.v))
      _Lfalse.append(lir.store(c, lir.ldcInsn(0, BooleanInfo, line.v), line.v))
      _Lfalse.append(lir.goto(_end, line.v))
      _v = lir.load(c, line.v)

      _entry = null
      _Ltrue = null
      _Lfalse = null

      _kind = CodeKind.V
    }
    assert(_kind == CodeKind.V)
    assert(_start != null)
  }

  private def checkC(): Unit = {
    if (_kind == null)
      _kind = CodeKind.C
    assert(_kind == CodeKind.C)
    assert(_entry != null)
  }

  def check(): Unit = {
    if (_kind == null || _kind == CodeKind.C)
      assert(_entry != null)
    else {
      assert(_kind == CodeKind.V)
      assert(_start != null)
    }
  }

  def clear(): Unit = {
    _entry = null
    _Ltrue = null
    _Lfalse = null
    _start = null
    _end = null
    _v = null
  }

  def unary_!(): CCode = {
    val newC = new CCode(entry, Lfalse, Ltrue)
    clear()
    newC
  }

  def &&(rhs: CCode)(implicit line: LineNumber): CCode = {
    Ltrue.append(lir.goto(rhs.entry, line.v))
    rhs.Lfalse.append(lir.goto(Lfalse, line.v))
    val newC = new CCode(entry, rhs.Ltrue, Lfalse)
    clear()
    rhs.clear()
    newC
  }

  def ||(rhs: CCode)(implicit line: LineNumber): CCode = {
    Lfalse.append(lir.goto(rhs.entry, line.v))
    rhs.Ltrue.append(lir.goto(Ltrue, line.v))
    val newC = new CCode(entry, Ltrue, rhs.Lfalse)
    clear()
    rhs.clear()
    newC
  }
}

class CodeBoolean(val lhs: Code[Boolean]) extends AnyVal {
  def toCCode(implicit line: LineNumber): CCode = lhs match {
    case x: CCode =>
      x
    case _ =>
      val Ltrue = new lir.Block()
      val Lfalse = new lir.Block()
      lhs.v match {
        case v: lir.LdcX =>
          val L = if (v.a.asInstanceOf[Int] != 0) Ltrue else Lfalse
          lhs.end.append(lir.goto(L, line.v))
        case _ =>
          assert(lhs.v.ti == BooleanInfo,lhs.v.ti)
          lhs.end.append(lir.ifx(IFNE, lhs.v, Ltrue, Lfalse, line.v))
      }
      val newC = new CCode(lhs.start, Ltrue, Lfalse)
      lhs.clear()
      newC
  }

  def unary_!()(implicit line: LineNumber): Code[Boolean] = !lhs.toCCode

  def muxAny(cthen: Code[_], celse: Code[_])(implicit line: LineNumber): Code[_] = {
    mux[Any](coerce[Any](cthen), coerce[Any](celse))
  }

  def mux[T](cthen: Code[T], celse: Code[T])(implicit line: LineNumber): Code[T] = {
    val cond = lhs.toCCode
    val L = new lir.Block()
    val newC = if (cthen.v == null) {
      assert(celse.v == null)

      cond.Ltrue.append(lir.goto(cthen.start, line.v))
      cthen.end.append(lir.goto(L, line.v))
      cond.Lfalse.append(lir.goto(celse.start, line.v))
      celse.end.append(lir.goto(L, line.v))
      new VCode(cond.entry, L, null)
    } else {
      assert(celse.v != null)
      assert(cthen.v.ti.desc == celse.v.ti.desc, s"${ cthen.v.ti.desc } == ${ celse.v.ti.desc }")

      val t = new lir.Local(null, "mux",
        cthen.v.ti)

      cond.Ltrue.append(lir.goto(cthen.start, line.v))
      cthen.end.append(lir.store(t, cthen.v, line.v))
      cthen.end.append(lir.goto(L, line.v))

      cond.Lfalse.append(lir.goto(celse.start, line.v))
      celse.end.append(lir.store(t, celse.v, line.v))
      celse.end.append(lir.goto(L, line.v))

      new VCode(cond.entry, L, lir.load(t, line.v))
    }
    cthen.clear()
    celse.clear()
    newC
  }

  def orEmpty(cthen: Code[Unit])(implicit line: LineNumber): Code[Unit] = {
    val cond = lhs.toCCode
    val L = new lir.Block()
    cond.Ltrue.append(lir.goto(cthen.start, line.v))
    cthen.end.append(lir.goto(L, line.v))
    cond.Lfalse.append(lir.goto(L, line.v))
    val newC = new VCode(cond.entry, L, null)
    cthen.clear()
    newC
  }

  def &(rhs: Code[Boolean])(implicit line: LineNumber): Code[Boolean] =
    Code(lhs, rhs, lir.insn2(IAND, line.v))

  def &&(rhs: Code[Boolean])(implicit line: LineNumber): Code[Boolean] =
    lhs.toCCode && rhs.toCCode

  def |(rhs: Code[Boolean])(implicit line: LineNumber): Code[Boolean] =
    Code(lhs, rhs, lir.insn2(IOR, line.v))

  def ||(rhs: Code[Boolean])(implicit line: LineNumber): Code[Boolean] =
    lhs.toCCode || rhs.toCCode

  def ceq(rhs: Code[Boolean])(implicit line: LineNumber): Code[Boolean] =
    lhs.toI.ceq(rhs.toI)

  def cne(rhs: Code[Boolean])(implicit line: LineNumber): Code[Boolean] =
    lhs.toI.cne(rhs.toI)

  // on the JVM Booleans are represented as Ints
  def toI: Code[Int] = lhs.asInstanceOf[Code[Int]]

  def toS(implicit line: LineNumber): Code[String] = lhs.mux(const("true"), const("false"))
}

class CodeInt(val lhs: Code[Int]) extends AnyVal {
  def unary_-()(implicit line: LineNumber): Code[Int] = Code(lhs, lir.insn1(INEG, line.v))

  def +(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IADD, line.v))

  def -(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(ISUB, line.v))

  def *(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IMUL, line.v))

  def /(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IDIV, line.v))

  def %(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IREM, line.v))

  def max(rhs: Code[Int])(implicit line: LineNumber): Code[Int] =
    Code.invokeStatic2[Math, Int, Int, Int]("max", lhs, rhs)

  def min(rhs: Code[Int])(implicit line: LineNumber): Code[Int] =
    Code.invokeStatic2[Math, Int, Int, Int]("min", lhs, rhs)

  def compare(op: Int, rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = {
    val Ltrue = new lir.Block()
    val Lfalse = new lir.Block()

    val entry = lhs.start
    lhs.end.append(lir.goto(rhs.start, line.v))
    rhs.end.append(lir.ifx(op, lhs.v, rhs.v, Ltrue, Lfalse, line.v))

    val newC = new CCode(entry, Ltrue, Lfalse)
    lhs.clear()
    rhs.clear()
    newC
  }

  def >(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.compare(IF_ICMPGT, rhs)

  def >=(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.compare(IF_ICMPGE, rhs)

  def <(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.compare(IF_ICMPLT, rhs)

  def <=(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.compare(IF_ICMPLE, rhs)

  def >>(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(ISHR, line.v))

  def <<(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(ISHL, line.v))

  def >>>(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IUSHR, line.v))

  def &(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IAND, line.v))

  def |(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IOR, line.v))

  def ^(rhs: Code[Int])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(IXOR, line.v))

  def unary_~()(implicit line: LineNumber): Code[Int] = lhs ^ const(-1)

  def ceq(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.compare(IF_ICMPEQ, rhs)

  def cne(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.compare(IF_ICMPNE, rhs)

  def toI: Code[Int] = lhs

  def toL(implicit line: LineNumber): Code[Long] = Code(lhs, lir.insn1(I2L, line.v))

  def toF(implicit line: LineNumber): Code[Float] = Code(lhs, lir.insn1(I2F, line.v))

  def toD(implicit line: LineNumber): Code[Double] = Code(lhs, lir.insn1(I2D, line.v))

  def toB(implicit line: LineNumber): Code[Byte] = Code(lhs, lir.insn1(I2B, line.v))

  // on the JVM Booleans are represented as Ints
  def toZ(implicit line: LineNumber): Code[Boolean] = lhs.cne(0)

  def toS(implicit line: LineNumber): Code[String] = Code.invokeStatic1[java.lang.Integer, Int, String]("toString", lhs)
}

class CodeLong(val lhs: Code[Long]) extends AnyVal {
  def unary_-()(implicit line: LineNumber): Code[Long] = Code(lhs, lir.insn1(LNEG, line.v))

  def +(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LADD, line.v))

  def -(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LSUB, line.v))

  def *(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LMUL, line.v))

  def /(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LDIV, line.v))

  def %(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LREM, line.v))

  def compare(rhs: Code[Long])(implicit line: LineNumber): Code[Int] = Code(lhs, rhs, lir.insn2(LCMP, line.v))

  def <(rhs: Code[Long])(implicit line: LineNumber): Code[Boolean] = compare(rhs) < 0

  def <=(rhs: Code[Long])(implicit line: LineNumber): Code[Boolean] = compare(rhs) <= 0

  def >(rhs: Code[Long])(implicit line: LineNumber): Code[Boolean] = compare(rhs) > 0

  def >=(rhs: Code[Long])(implicit line: LineNumber): Code[Boolean] = compare(rhs) >= 0

  def ceq(rhs: Code[Long])(implicit line: LineNumber): Code[Boolean] = compare(rhs) ceq 0

  def cne(rhs: Code[Long])(implicit line: LineNumber): Code[Boolean] = compare(rhs) cne 0

  def >>(rhs: Code[Int])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LSHR, line.v))

  def <<(rhs: Code[Int])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LSHL, line.v))

  def >>>(rhs: Code[Int])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LUSHR, line.v))

  def &(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LAND, line.v))

  def |(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LOR, line.v))

  def ^(rhs: Code[Long])(implicit line: LineNumber): Code[Long] = Code(lhs, rhs, lir.insn2(LXOR, line.v))

  def unary_~()(implicit line: LineNumber): Code[Long] = lhs ^ const(-1L)

  def toI(implicit line: LineNumber): Code[Int] = Code(lhs, lir.insn1(L2I, line.v))

  def toL: Code[Long] = lhs

  def toF(implicit line: LineNumber): Code[Float] = Code(lhs, lir.insn1(L2F, line.v))

  def toD(implicit line: LineNumber): Code[Double] = Code(lhs, lir.insn1(L2D, line.v))

  def toS(implicit line: LineNumber): Code[String] = Code.invokeStatic1[java.lang.Long, Long, String]("toString", lhs)
}

class CodeFloat(val lhs: Code[Float]) extends AnyVal {
  def unary_-()(implicit line: LineNumber): Code[Float] = Code(lhs, lir.insn1(FNEG, line.v))

  def +(rhs: Code[Float])(implicit line: LineNumber): Code[Float] = Code(lhs, rhs, lir.insn2(FADD, line.v))

  def -(rhs: Code[Float])(implicit line: LineNumber): Code[Float] = Code(lhs, rhs, lir.insn2(FSUB, line.v))

  def *(rhs: Code[Float])(implicit line: LineNumber): Code[Float] = Code(lhs, rhs, lir.insn2(FMUL, line.v))

  def /(rhs: Code[Float])(implicit line: LineNumber): Code[Float] = Code(lhs, rhs, lir.insn2(FDIV, line.v))

  def >(rhs: Code[Float])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(FCMPL, line.v)) > 0

  def >=(rhs: Code[Float])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(FCMPL, line.v)) >= 0

  def <(rhs: Code[Float])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(FCMPG, line.v)) < 0

  def <=(rhs: Code[Float])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(FCMPG, line.v)) <= 0

  def ceq(rhs: Code[Float])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(FCMPL, line.v)).ceq(0)

  def cne(rhs: Code[Float])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(FCMPL, line.v)).cne(0)

  def toI(implicit line: LineNumber): Code[Int] = Code(lhs, lir.insn1(F2I, line.v))

  def toL(implicit line: LineNumber): Code[Long] = Code(lhs, lir.insn1(F2L, line.v))

  def toF: Code[Float] = lhs

  def toD(implicit line: LineNumber): Code[Double] = Code(lhs, lir.insn1(F2D, line.v))

  def toS(implicit line: LineNumber): Code[String] = Code.invokeStatic1[java.lang.Float, Float, String]("toString", lhs)
}

class CodeDouble(val lhs: Code[Double]) extends AnyVal {
  def unary_-()(implicit line: LineNumber): Code[Double] = Code(lhs, lir.insn1(DNEG, line.v))

  def +(rhs: Code[Double])(implicit line: LineNumber): Code[Double] = Code(lhs, rhs, lir.insn2(DADD, line.v))

  def -(rhs: Code[Double])(implicit line: LineNumber): Code[Double] = Code(lhs, rhs, lir.insn2(DSUB, line.v))

  def *(rhs: Code[Double])(implicit line: LineNumber): Code[Double] = Code(lhs, rhs, lir.insn2(DMUL, line.v))

  def /(rhs: Code[Double])(implicit line: LineNumber): Code[Double] = Code(lhs, rhs, lir.insn2(DDIV, line.v))

  def >(rhs: Code[Double])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(DCMPL, line.v)) > 0

  def >=(rhs: Code[Double])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(DCMPL, line.v)) >= 0

  def <(rhs: Code[Double])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(DCMPG, line.v)) < 0

  def <=(rhs: Code[Double])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(DCMPG, line.v)) <= 0

  def ceq(rhs: Code[Double])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(DCMPL, line.v)).ceq(0)

  def cne(rhs: Code[Double])(implicit line: LineNumber): Code[Boolean] = Code[Int](lhs, rhs, lir.insn2(DCMPL, line.v)).cne(0)

  def toI(implicit line: LineNumber): Code[Int] = Code(lhs, lir.insn1(D2I, line.v))

  def toL(implicit line: LineNumber): Code[Long] = Code(lhs, lir.insn1(D2L, line.v))

  def toF(implicit line: LineNumber): Code[Float] = Code(lhs, lir.insn1(D2F, line.v))

  def toD: Code[Double] = lhs

  def toS(implicit line: LineNumber): Code[String] = Code.invokeStatic1[java.lang.Double, Double, String]("toString", lhs)
}

class CodeChar(val lhs: Code[Char]) extends AnyVal {
  def +(rhs: Code[Char])(implicit line: LineNumber): Code[Char] = Code(lhs, rhs, lir.insn2(IADD, line.v))

  def -(rhs: Code[Char])(implicit line: LineNumber): Code[Char] = Code(lhs, rhs, lir.insn2(ISUB, line.v))

  def >(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.toI > rhs.toI

  def >=(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.toI >= rhs.toI

  def <(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.toI < rhs.toI

  def <=(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.toI <= rhs

  def ceq(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.toI.ceq(rhs)

  def cne(rhs: Code[Int])(implicit line: LineNumber): Code[Boolean] = lhs.toI.cne(rhs)

  def toI: Code[Int] = lhs.asInstanceOf[Code[Int]]

  def toS(implicit line: LineNumber): Code[String] = Code.invokeStatic1[java.lang.String, Char, String]("valueOf", lhs)
}

class CodeString(val lhs: Code[String]) extends AnyVal {
  def concat(other: Code[String])(implicit line: LineNumber): Code[String] = lhs.invoke[String, String]("concat", other)

  def println()(implicit line: LineNumber): Code[Unit] = Code.getStatic[System, PrintStream]("out").invoke[String, Unit]("println", lhs)

  def length()(implicit line: LineNumber): Code[Int] = lhs.invoke[Int]("length")

  def apply(i: Code[Int])(implicit line: LineNumber): Code[Char] = lhs.invoke[Int, Char]("charAt", i)
}

class CodeArray[T](val lhs: Code[Array[T]])(implicit tti: TypeInfo[T]) {
  def apply(i: Code[Int])(implicit line: LineNumber): Code[T] =
    Code(lhs, i, lir.insn2(tti.aloadOp, line.v))

  def update(i: Code[Int], x: Code[T])(implicit line: LineNumber): Code[Unit] = {
    lhs.start.append(lir.goto(i.end, line.v))
    i.start.append(lir.goto(x.start, line.v))
    x.end.append(lir.stmtOp(tti.astoreOp, lhs.v, i.v, x.v, line.v))
    val newC = new VCode(lhs.start, x.end, null)
    lhs.clear()
    i.clear()
    x.clear()
    newC
  }

  def length()(implicit line: LineNumber): Code[Int] =
    Code(lhs, lir.insn1(ARRAYLENGTH, line.v))
}

object CodeLabel {
  def apply(): CodeLabel = {
    val L = new lir.Block()
    new CodeLabel(L)
  }
}

class CodeLabel(val L: lir.Block) extends Code[Unit] {
  private var _start: lir.Block = L

  def start: lir.Block = {
    check()
    _start
  }

  def end: lir.Block = {
    check()
    _start
  }

  def v: lir.ValueX = {
    check()
    null
  }

  def check(): Unit = {
    assert(_start != null)
  }

  def clear(): Unit = {
    _start = null
  }

  def goto(implicit line: LineNumber): Code[Unit] = {
    val M = new lir.Block()
    M.append(lir.goto(L, line.v))
    new VCode(M, M, null)
  }
}

object Invokeable {
  def apply[T](cls: Class[T], c: reflect.Constructor[_]): Invokeable[T, Unit] = new Invokeable[T, Unit](
    cls,
    "<init>",
    isStatic = false,
    isInterface = false,
    INVOKESPECIAL,
    Type.getConstructorDescriptor(c),
    implicitly[ClassTag[Unit]].runtimeClass)

  def apply[T, S](cls: Class[T], m: reflect.Method)(implicit sct: ClassTag[S]): Invokeable[T, S] = {
    val isInterface = m.getDeclaringClass.isInterface
    val isStatic = reflect.Modifier.isStatic(m.getModifiers)
    assert(!(isInterface && isStatic))
    new Invokeable[T, S](cls,
      m.getName,
      isStatic,
      isInterface,
      if (isInterface)
        INVOKEINTERFACE
      else if (isStatic)
        INVOKESTATIC
      else
        INVOKEVIRTUAL,
      Type.getMethodDescriptor(m),
      m.getReturnType)
  }

  def lookupMethod[T, S](cls: Class[T], method: String, parameterTypes: Array[Class[_]])(implicit sct: ClassTag[S]): Invokeable[T, S] = {
    val m = cls.getMethod(method, parameterTypes: _*)
    assert(m != null,
      s"no such method ${ cls.getName }.$method(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    // generic type parameters return java.lang.Object instead of the correct class
    assert(m.getReturnType.isAssignableFrom(sct.runtimeClass),
      s"when invoking ${ cls.getName }.$method(): ${ m.getReturnType.getName }: wrong return type ${ sct.runtimeClass.getName }")

    Invokeable(cls, m)
  }
}

class Invokeable[T, S](tcls: Class[T],
  val name: String,
  val isStatic: Boolean,
  val isInterface: Boolean,
  val invokeOp: Int,
  val descriptor: String,
  val concreteReturnType: Class[_])(implicit sct: ClassTag[S]) {
  def invoke(lhs: Code[T], args: Array[Code[_]])(implicit line: LineNumber): Code[S] = {
    val (start, end, argvs) = Code.sequenceValues(
      if (isStatic)
        args
      else
        lhs +: args)

    val sti = typeInfoFromClassTag(sct)

    if (sct.runtimeClass == java.lang.Void.TYPE) {
      end.append(
        lir.methodStmt(invokeOp, Type.getInternalName(tcls), name, descriptor, isInterface, sti, argvs, line.v))
      new VCode(start, end, null)
    } else {
      val t = new lir.Local(null, "invoke", sti)
      var r = lir.methodInsn(invokeOp, Type.getInternalName(tcls), name, descriptor, isInterface, sti, argvs, line.v)
      if (concreteReturnType != sct.runtimeClass)
        r = lir.checkcast(Type.getInternalName(sct.runtimeClass), r, line.v)
      end.append(lir.store(t, r, line.v))
      new VCode(start, end, lir.load(t, line.v))
    }
  }
}

object FieldRef {
  def apply[T, S](field: String)(implicit tct: ClassTag[T], sct: ClassTag[S], sti: TypeInfo[S]): FieldRef[T, S] = {
    val f = tct.runtimeClass.getDeclaredField(field)
    assert(f.getType == sct.runtimeClass,
      s"when getting field ${ tct.runtimeClass.getName }.$field: ${ f.getType.getName }: wrong type ${ sct.runtimeClass.getName } ")

    new FieldRef(f)
  }
}

trait Value[+T] { self =>
  def get(implicit line: LineNumber): Code[T]
}

trait Settable[T] extends Value[T] {
  def store(rhs: Code[T])(implicit line: LineNumber): Code[Unit]

  def :=(rhs: Code[T])(implicit line: LineNumber): Code[Unit] = store(rhs)

  def storeAny(rhs: Code[_])(implicit line: LineNumber): Code[Unit] = store(coerce[T](rhs))

  def load()(implicit line: LineNumber): Code[T] = get
}

class ThisLazyFieldRef[T: TypeInfo](cb: ClassBuilder[_], name: String, setup: Code[T])(implicit line: LineNumber) extends Value[T] {
  private[this] val value: Settable[T] = cb.genFieldThisRef[T](name)
  private[this] val present: Settable[Boolean] = cb.genFieldThisRef[Boolean](s"${name}_present")

  private[this] val setm = cb.genMethod[Unit](s"setup_$name")
  setm.emit(Code(value := setup, present := true))

  def get(implicit line: LineNumber): Code[T] =
    Code(present.mux(Code._empty, setm.invoke()), value.load())
}

class ThisFieldRef[T: TypeInfo](cb: ClassBuilder[_], f: Field[T]) extends Settable[T] {
  def name: String = f.name

  def get(implicit line: LineNumber): Code[T] = f.get(cb._this)

  def store(rhs: Code[T])(implicit line: LineNumber): Code[Unit] = f.put(cb._this, rhs)
}

class StaticFieldRef[T: TypeInfo](f: StaticField[T]) extends Settable[T] {
  def name: String = f.name

  def get(implicit line: LineNumber): Code[T] = f.get()

  def store(rhs: Code[T])(implicit line: LineNumber): Code[Unit] = f.put(rhs)
}

class LocalRef[T](val l: lir.Local) extends Settable[T] {
  def get(implicit line: LineNumber): Code[T] = Code(lir.load(l, line.v))

  def store(rhs: Code[T])(implicit line: LineNumber): Code[Unit] = {
    assert(rhs.v != null)
    rhs.end.append(lir.store(l, rhs.v, line.v))
    val newC = new VCode(rhs.start, rhs.end, null)
    rhs.clear()
    newC
  }
}

class LocalRefInt(val v: LocalRef[Int]) extends AnyRef {
  def +=(i: Int)(implicit line: LineNumber): Code[Unit] = {
    val L = new lir.Block()
    L.append(lir.iincInsn(v.l, i, line.v))
    new VCode(L, L, null)
  }

  def ++()(implicit line: LineNumber): Code[Unit] = +=(1)
}

class FieldRef[T, S](f: reflect.Field)(implicit tct: ClassTag[T], sti: TypeInfo[S]) {
  self =>

  val tiname = Type.getInternalName(tct.runtimeClass)

  def isStatic: Boolean = reflect.Modifier.isStatic(f.getModifiers)

  def getOp = if (isStatic) GETSTATIC else GETFIELD

  def putOp = if (isStatic) PUTSTATIC else PUTFIELD

  def getField()(implicit line: LineNumber): Code[S] = getField(null: Value[T])

  def getField(lhs: Value[T]): Value[S] =
    new Value[S] {
      def get(implicit line: LineNumber): Code[S] = self.getField(if (lhs != null) lhs.get else null)
    }

  def getField(lhs: Code[T])(implicit line: LineNumber): Code[S] =
    if (isStatic)
      Code(lir.getStaticField(tiname, f.getName, sti, line.v))
    else
      Code(lhs, lir.getField(tiname, f.getName, sti, line.v))

  def put(lhs: Code[T], rhs: Code[S])(implicit line: LineNumber): Code[Unit] =
    if (isStatic)
      Code.void(rhs, lir.putStaticField(tiname, f.getName, sti, line.v))
    else
      Code.void(lhs, rhs, lir.putField(tiname, f.getName, sti, line.v))
}

class CodeObject[T <: AnyRef : ClassTag](val lhs: Code[T]) {
  def getField[S](field: String)(implicit sct: ClassTag[S], sti: TypeInfo[S], line: LineNumber): Code[S] =
    FieldRef[T, S](field).getField(lhs)

  def put[S](field: String, rhs: Code[S])(implicit sct: ClassTag[S], sti: TypeInfo[S], line: LineNumber): Code[Unit] =
    FieldRef[T, S](field).put(lhs, rhs)

  def invoke[S](method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]]
  )(implicit sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    Invokeable.lookupMethod[T, S](implicitly[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]], method, parameterTypes).invoke(lhs, args)

  def invoke[S](method: String)(implicit sct: ClassTag[S], line: LineNumber): Code[S] =
    invoke[S](method, Array[Class[_]](), Array[Code[_]]())

  def invoke[A1, S](method: String, a1: Code[A1]
  )(implicit a1ct: ClassTag[A1], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def invoke[A1, A2, S](method: String, a1: Code[A1], a2: Code[A2]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def invoke[A1, A2, A3, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3))

  def invoke[A1, A2, A3, A4, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4))

  def invoke[A1, A2, A3, A4, A5, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5))

  def invoke[A1, A2, A3, A4, A5, A6, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], sct: ClassTag[S], line: LineNumber
  ): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5, a6))

  def invoke[A1, A2, A3, A4, A5, A6, A7, A8, S](method: String, a1: Code[A1], a2: Code[A2],
    a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6], a7: Code[A7], a8: Code[A8]
  )(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5],
    a6ct: ClassTag[A6], a7ct: ClassTag[A7], a8ct: ClassTag[A8], sct: ClassTag[S], line: LineNumber
  ): Code[S] = {
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass,
      a6ct.runtimeClass, a7ct.runtimeClass, a8ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5, a6, a7, a8))
  }
}

class CodeNullable[T >: Null : TypeInfo](val lhs: Code[T]) {
  def isNull(implicit line: LineNumber): Code[Boolean] = {
    val Ltrue = new lir.Block()
    val Lfalse = new lir.Block()

    val entry = lhs.start
    lhs.end.append(lir.ifx(IFNULL, lhs.v, Ltrue, Lfalse, line.v))

    val newC = new CCode(entry, Ltrue, Lfalse)
    lhs.clear()
    newC
  }

  def ifNull[U](cnullcase: Code[U], cnonnullcase: Code[U])(implicit line: LineNumber): Code[U] =
    isNull.mux(cnullcase, cnonnullcase)

  def mapNull[U >: Null](cnonnullcase: Code[U])(implicit uti: TypeInfo[U], line: LineNumber): Code[U] =
    ifNull[U](Code._null[U], cnonnullcase)
}
