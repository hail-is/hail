package is.hail.asm4s

import java.io.PrintStream

import is.hail.expr.CM
import java.lang.reflect.{Constructor, Field, Method, Modifier}
import java.util

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.reflect.ClassTag

object Code {
  def apply[T](insn: => AbstractInsnNode): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      il += insn
    }
  }

  def apply[S1, S2](c1: Code[S1], c2: Code[S2]): Code[S2] =
    new Code[S2] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
      }
    }

  def apply[S1, S2, S3](c1: Code[S1], c2: Code[S2], c3: Code[S3]): Code[S3] =
    new Code[S3] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
      }
    }

  def apply[S1, S2, S3, S4](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4]): Code[S4] =
    new Code[S4] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5]): Code[S5] =
    new Code[S5] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5, S6](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5], c6: Code[S6]): Code[S6] =
    new Code[S6] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
        c6.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5, S6, S7](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5], c6: Code[S6], c7: Code[S7]): Code[S7] =
    new Code[S7] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
        c6.emit(il)
        c7.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5, S6, S7, S8](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5], c6: Code[S6], c7: Code[S7], c8: Code[S8]): Code[S8] =
    new Code[S8] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
        c6.emit(il)
        c7.emit(il)
        c8.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5, S6, S7, S8, S9](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5], c6: Code[S6], c7: Code[S7], c8: Code[S8], c9: Code[S9]): Code[S9] =
    new Code[S9] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
        c6.emit(il)
        c7.emit(il)
        c8.emit(il)
        c9.emit(il)
      }
    }

  def apply(cs: Code[_]*): Code[_] =
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        cs.foreach(_.emit(il))
      }
    }

  def newInstance[T](parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit tct: ClassTag[T], tti: TypeInfo[T]): Code[T] = {
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        il += new TypeInsnNode(NEW, Type.getInternalName(tct.runtimeClass))
        il += new InsnNode(DUP)
        Invokeable.lookupConstructor[T](tct.runtimeClass.asInstanceOf[Class[T]], parameterTypes).invoke(null, args).emit(il)
      }
    }
  }

  def newInstance[T]()(implicit tct: ClassTag[T], tti: TypeInfo[T]): Code[T] =
    newInstance[T](Array[Class[_]](), Array[Code[_]]())

  def newInstance[T, A1](a1: Code[A1])(implicit a1ct: ClassTag[A1],
    tct: ClassTag[T], tti: TypeInfo[T]): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def newInstance[T, A1, A2](a1: Code[A1], a2: Code[A2])(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2],
    tct: ClassTag[T], tti: TypeInfo[T]): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def newInstance[T, A1, A2, A3](a1: Code[A1], a2: Code[A2], a3: Code[A3])(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2],
    a3ct: ClassTag[A3], tct: ClassTag[T], tti: TypeInfo[T]): Code[T] =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3))

  def newArray[T](size: Code[Int])(implicit tti: TypeInfo[T]): Code[Array[T]] = {
    new Code[Array[T]] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        size.emit(il)
        il += tti.newArray()
      }
    }
  }

  def whileLoop(condition: Code[Boolean], body: Code[_]*): Code[Unit] = {
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        val l1 = new LabelNode
        val l2 = new LabelNode
        val l3 = new LabelNode
        il += l1
        condition.toConditional.emitConditional(il, l2, l3)
        il += l2
        body.foreach(_.emit(il))
        il += new JumpInsnNode(GOTO, l1)
        il += l3
      }
    }
  }

  def invokeScalaObject[S](cls: Class[_], method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit sct: ClassTag[S]): Code[S] = {
    val m = Invokeable.lookupMethod(cls, method, parameterTypes)(sct)
    val staticObj = FieldRef("MODULE$")(ClassTag(cls), ClassTag(cls), classInfo(ClassTag(cls)))
    m.invoke(staticObj.get(), args)
  }

  def invokeScalaObject[A1, S](cls: Class[_], method: String, a1: Code[A1])(implicit a1ct: ClassTag[A1], sct: ClassTag[S]): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass), Array(a1))

  def invokeScalaObject[A1, A2, S](cls: Class[_], method: String, a1: Code[A1], a2: Code[A2])(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], sct: ClassTag[S]): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array(a1, a2))

  def invokeScalaObject[A1, A2, A3, S](cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3])(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], sct: ClassTag[S]): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array(a1, a2, a3))

  def invokeScalaObject[A1, A2, A3, A4, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4])(
    implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], sct: ClassTag[S]): Code[S] =
    invokeScalaObject[S](cls, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array(a1, a2, a3, a4))

  def invokeScalaObject[A1, A2, A3, A4, A5, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5])(
    implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], sct: ClassTag[S]
  ): Code[S] =
    invokeScalaObject[S](
      cls, method, Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass), Array(a1, a2, a3, a4, a5))

  def invokeScalaObject[A1, A2, A3, A4, A5, A6, S](
    cls: Class[_], method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5], a6: Code[A6])(
    implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], a6ct: ClassTag[A6], sct: ClassTag[S]
  ): Code[S] =
    invokeScalaObject[S](
      cls, method, Array[Class[_]](
        a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass, a6ct.runtimeClass), Array(a1, a2, a3, a4, a5, a6))

  def invokeStatic[S](cls: Class[_], method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit sct: ClassTag[S]): Code[S] = {
    val m = Invokeable.lookupMethod(cls, method, parameterTypes)(sct)
    assert(m.isStatic)
    m.invoke(null, args)
  }

  def invokeStatic[T, S](method: String)(implicit tct: ClassTag[T], sct: ClassTag[S]): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](), Array[Code[_]]())

  def invokeStatic[T, A1, S](method: String, a1: Code[A1])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1]): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))(sct)

  def invokeStatic[T, A1, A2, S](method: String, a1: Code[A1], a2: Code[A2])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2]): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))(sct)

  def invokeStatic[T, A1, A2, A3, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3]): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3))(sct)

  def invokeStatic[T, A1, A2, A3, A4, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4]): Code[S] =
    invokeStatic[S](tct.runtimeClass, method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4))(sct)

  def _null[T >: Null]: Code[T] = Code(new InsnNode(ACONST_NULL))

  def toUnit[T](c: Code[T])(implicit tti: TypeInfo[T]): Code[Unit] = {
    val op = tti.slots match {
      case 1 => POP
      case 2 => POP2
    }
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        c.emit(il)
        il += new InsnNode(op)
      }
    }
  }

  // FIXME: code should really carry around the stack so this type can be correct
  // Currently, this is a huge potential place for errors.
  def _empty[T]: Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
    }
  }

  def _throw[T <: java.lang.Throwable, U](cerr: Code[T]): Code[U] = Code(cerr, new InsnNode(ATHROW))

  def _fatal[U](msg: Code[String]): Code[U] =
    Code._throw[is.hail.utils.HailException, U](Code.newInstance[is.hail.utils.HailException, String, Option[String], Throwable](
      msg,
      Code.invokeStatic[scala.Option[String], scala.Option[String]]("empty"),
      Code._null[Throwable]))

  def _return[T](c: Code[T])(implicit tti: TypeInfo[T]): Code[Unit] =
    Code(c, Code(new InsnNode(tti.returnOp)))

  def checkcast[T](v: Code[AnyRef])(implicit tct: ClassTag[T]): Code[T] = Code(
    v,
    new TypeInsnNode(CHECKCAST, Type.getInternalName(tct.runtimeClass)))

  def boxBoolean(cb: Code[Boolean]): Code[java.lang.Boolean] = Code.newInstance[java.lang.Boolean, Boolean](cb)

  def boxInt(ci: Code[Int]): Code[java.lang.Integer] = Code.newInstance[java.lang.Integer, Int](ci)

  def boxLong(cl: Code[Long]): Code[java.lang.Long] = Code.newInstance[java.lang.Long, Long](cl)

  def boxFloat(cf: Code[Float]): Code[java.lang.Float] = Code.newInstance[java.lang.Float, Float](cf)

  def boxDouble(cd: Code[Double]): Code[java.lang.Double] = Code.newInstance[java.lang.Double, Double](cd)

  def booleanValue(x: Code[java.lang.Boolean]): Code[Boolean] = x.invoke[Boolean]("booleanValue")

  def intValue(x: Code[java.lang.Number]): Code[Int] = x.invoke[Int]("intValue")

  def longValue(x: Code[java.lang.Number]): Code[Long] = x.invoke[Long]("longValue")

  def floatValue(x: Code[java.lang.Number]): Code[Float] = x.invoke[Float]("floatValue")

  def doubleValue(x: Code[java.lang.Number]): Code[Double] = x.invoke[Double]("doubleValue")

  def getStatic[T: ClassTag, S: ClassTag : TypeInfo](field: String): Code[S] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.get(null)
  }

  def putStatic[T: ClassTag, S: ClassTag : TypeInfo](field: String, rhs: Code[S]): Code[Unit] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.put(null, rhs)
  }
}

trait Code[+T] {
  self =>
  def emit(il: Growable[AbstractInsnNode]): Unit

  def compare[U >: T](opcode: Int, rhs: Code[U]): CodeConditional =
    new CodeConditional {
      def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) {
        self.emit(il)
        rhs.emit(il)
        il += new JumpInsnNode(opcode, ltrue)
        il += new JumpInsnNode(GOTO, lfalse)
      }
    }
}

trait CodeConditional extends Code[Boolean] {
  self =>
  def emit(il: Growable[AbstractInsnNode]): Unit = {
    val lafter = new LabelNode
    val ltrue = new LabelNode
    val lfalse = new LabelNode
    emitConditional(il, ltrue, lfalse)
    il += lfalse
    il += new LdcInsnNode(0)
    il += new JumpInsnNode(GOTO, lafter)
    il += ltrue
    il += new LdcInsnNode(1)
    il += lafter
  }

  def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode): Unit

  def unary_!(): CodeConditional =
    new CodeConditional {
      def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) {
        self.emitConditional(il, lfalse, ltrue)
      }
    }

  def &&(rhs: CodeConditional) = new CodeConditional {
    def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) = {
      val lt2 = new LabelNode
      self.emitConditional(il, lt2, lfalse)
      il += lt2
      rhs.emitConditional(il, ltrue, lfalse)
    }
  }

  def ||(rhs: CodeConditional) = new CodeConditional {
    def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) = {
      val lf2 = new LabelNode
      self.emitConditional(il, ltrue, lf2)
      il += lf2
      rhs.emitConditional(il, ltrue, lfalse)
    }
  }

  def ceq(rhs: CodeConditional) = new CodeConditional {
    def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) = {
      val lefttrue = new LabelNode
      val leftfalse = new LabelNode
      self.emitConditional(il, lefttrue, leftfalse)
      il += lefttrue
      rhs.emitConditional(il, ltrue, lfalse)
      il += leftfalse
      rhs.emitConditional(il, lfalse, ltrue)
    }
  }

  def cne(rhs: CodeConditional) = new CodeConditional {
    def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) = {
      val lefttrue = new LabelNode
      val leftfalse = new LabelNode
      self.emitConditional(il, lefttrue, leftfalse)
      il += lefttrue
      rhs.emitConditional(il, lfalse, ltrue)
      il += leftfalse
      rhs.emitConditional(il, ltrue, lfalse)
    }
  }
}

class CodeBoolean(val lhs: Code[Boolean]) extends AnyVal {
  def toConditional: CodeConditional = lhs match {
    case cond: CodeConditional =>
      cond

    case _ =>
      new CodeConditional {
        def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode) {
          lhs.emit(il)
          il += new JumpInsnNode(IFEQ, lfalse)
          il += new JumpInsnNode(GOTO, ltrue)
        }
      }
  }

  def unary_!(): Code[Boolean] =
    !lhs.toConditional

  def mux[T](cthen: Code[T], celse: Code[T]): Code[T] = {
    val cond = lhs.toConditional
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        val lafter = new LabelNode
        val ltrue = new LabelNode
        val lfalse = new LabelNode
        cond.emitConditional(il, ltrue, lfalse)
        il += lfalse
        celse.emit(il)
        il += new JumpInsnNode(GOTO, lafter)
        il += ltrue
        cthen.emit(il)
        // fall through
        il += lafter
      }
    }
  }

  def &(rhs: Code[Boolean]): Code[Boolean] =
    Code(lhs, rhs, new InsnNode(IAND))

  def &&(rhs: Code[Boolean]): Code[Boolean] = {
    lhs.toConditional && rhs.toConditional
  }

  def |(rhs: Code[Boolean]): Code[Boolean] =
    Code(lhs, rhs, new InsnNode(IOR))

  def ||(rhs: Code[Boolean]): Code[Boolean] =
    lhs.toConditional || rhs.toConditional

  def ceq(rhs: Code[Boolean]): Code[Boolean] =
    lhs.toConditional.ceq(rhs.toConditional)

  def cne(rhs: Code[Boolean]): Code[Boolean] =
    lhs.toConditional.cne(rhs.toConditional)

  // on the JVM Booleans are represented as Ints
  def toI: Code[Int] = lhs.asInstanceOf[Code[Int]]

  def toS: Code[String] = lhs.mux(const("true"), const("false"))
}

class CodeInt(val lhs: Code[Int]) extends AnyVal {
  def unary_-(): Code[Int] = Code(lhs, new InsnNode(INEG))

  def +(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IADD))

  def -(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(ISUB))

  def *(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IMUL))

  def /(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IDIV))

  def >(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPGT, rhs)

  def >=(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPGE, rhs)

  def <(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPLT, rhs)

  def <=(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPLE, rhs)

  def >>(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(ISHR))

  def <<(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(ISHL))

  def >>>(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IUSHR))

  def &(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IAND))

  def |(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IOR))

  def ^(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IXOR))

  def unary_~(): Code[Int] = lhs ^ const(-1)

  def ceq(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPEQ, rhs)

  def cne(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPNE, rhs)

  def toI: Code[Int] = lhs

  def toL: Code[Long] = Code(lhs, new InsnNode(I2L))

  def toF: Code[Float] = Code(lhs, new InsnNode(I2F))

  def toD: Code[Double] = Code(lhs, new InsnNode(I2D))

  def toB: Code[Byte] = Code(lhs, new InsnNode(I2B))

  // on the JVM Booleans are represented as Ints
  def toZ: Code[Boolean] = lhs.asInstanceOf[Code[Boolean]]

  def toS: Code[String] = Code.invokeStatic[java.lang.Integer, Int, String]("toString", lhs)
}

class CodeLong(val lhs: Code[Long]) extends AnyVal {
  def unary_-(): Code[Long] = Code(lhs, new InsnNode(LNEG))

  def +(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LADD))

  def -(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LSUB))

  def *(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LMUL))

  def /(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LDIV))

  def compare(rhs: Code[Long]): Code[Int] = Code(lhs, rhs, new InsnNode(LCMP))

  def <(rhs: Code[Long]): Code[Boolean] = compare(rhs) < 0

  def <=(rhs: Code[Long]): Code[Boolean] = compare(rhs) <= 0

  def >(rhs: Code[Long]): Code[Boolean] = compare(rhs) > 0

  def >=(rhs: Code[Long]): Code[Boolean] = compare(rhs) >= 0

  def ceq(rhs: Code[Long]): Code[Boolean] = compare(rhs) ceq 0

  def cne(rhs: Code[Long]): Code[Boolean] = compare(rhs) cne 0

  def >>(rhs: Code[Int]): Code[Long] = Code(lhs, rhs, new InsnNode(LSHR))

  def <<(rhs: Code[Int]): Code[Long] = Code(lhs, rhs, new InsnNode(LSHL))

  def >>>(rhs: Code[Int]): Code[Long] = Code(lhs, rhs, new InsnNode(LUSHR))

  def &(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LAND))

  def |(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LOR))

  def ^(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LXOR))

  def unary_~(): Code[Long] = lhs ^ const(-1L)

  def toI: Code[Int] = Code(lhs, new InsnNode(L2I))

  def toL: Code[Long] = lhs

  def toF: Code[Float] = Code(lhs, new InsnNode(L2F))

  def toD: Code[Double] = Code(lhs, new InsnNode(L2D))

  def toS: Code[String] = Code.invokeStatic[java.lang.Long, Long, String]("toString", lhs)
}

class CodeFloat(val lhs: Code[Float]) extends AnyVal {
  def unary_-(): Code[Float] = Code(lhs, new InsnNode(FNEG))

  def +(rhs: Code[Float]): Code[Float] = Code(lhs, rhs, new InsnNode(FADD))

  def -(rhs: Code[Float]): Code[Float] = Code(lhs, rhs, new InsnNode(FSUB))

  def *(rhs: Code[Float]): Code[Float] = Code(lhs, rhs, new InsnNode(FMUL))

  def /(rhs: Code[Float]): Code[Float] = Code(lhs, rhs, new InsnNode(FDIV))

  def >(rhs: Code[Float]): Code[Boolean] = Code[Float, Float, Int](lhs, rhs, new InsnNode(FCMPL)) > 0

  def >=(rhs: Code[Float]): Code[Boolean] = Code[Float, Float, Int](lhs, rhs, new InsnNode(FCMPL)) >= 0

  def <(rhs: Code[Float]): Code[Boolean] = Code[Float, Float, Int](lhs, rhs, new InsnNode(FCMPG)) < 0

  def <=(rhs: Code[Float]): Code[Boolean] = Code[Float, Float, Int](lhs, rhs, new InsnNode(FCMPG)) <= 0

  def ceq(rhs: Code[Float]): Code[Boolean] = Code[Float, Float, Int](lhs, rhs, new InsnNode(FCMPL)).ceq(0)

  def cne(rhs: Code[Float]): Code[Boolean] = Code[Float, Float, Int](lhs, rhs, new InsnNode(FCMPL)).cne(0)

  def toI: Code[Int] = Code(lhs, new InsnNode(F2I))

  def toL: Code[Long] = Code(lhs, new InsnNode(F2L))

  def toF: Code[Float] = lhs

  def toD: Code[Double] = Code(lhs, new InsnNode(F2D))

  def toS: Code[String] = Code.invokeStatic[java.lang.Float, Float, String]("toString", lhs)
}

class CodeDouble(val lhs: Code[Double]) extends AnyVal {
  def unary_-(): Code[Double] = Code(lhs, new InsnNode(DNEG))

  def +(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DADD))

  def -(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DSUB))

  def *(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DMUL))

  def /(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DDIV))

  def >(rhs: Code[Double]): Code[Boolean] = Code[Double, Double, Int](lhs, rhs, new InsnNode(DCMPL)) > 0

  def >=(rhs: Code[Double]): Code[Boolean] = Code[Double, Double, Int](lhs, rhs, new InsnNode(DCMPL)) >= 0

  def <(rhs: Code[Double]): Code[Boolean] = Code[Double, Double, Int](lhs, rhs, new InsnNode(DCMPG)) < 0

  def <=(rhs: Code[Double]): Code[Boolean] = Code[Double, Double, Int](lhs, rhs, new InsnNode(DCMPG)) <= 0

  def ceq(rhs: Code[Double]): Code[Boolean] = Code[Double, Double, Int](lhs, rhs, new InsnNode(DCMPL)).ceq(0)

  def cne(rhs: Code[Double]): Code[Boolean] = Code[Double, Double, Int](lhs, rhs, new InsnNode(DCMPL)).cne(0)

  def toI: Code[Int] = Code(lhs, new InsnNode(D2I))

  def toL: Code[Long] = Code(lhs, new InsnNode(D2L))

  def toF: Code[Float] = Code(lhs, new InsnNode(D2F))

  def toD: Code[Double] = lhs

  def toS: Code[String] = Code.invokeStatic[java.lang.Double, Double, String]("toString", lhs)
}

class CodeString(val lhs: Code[String]) extends AnyVal {
  def concat(other: Code[String]): Code[String] = lhs.invoke[String, String]("concat", other)

  def println(): Code[Unit] = Code.getStatic[System, PrintStream]("out").invoke[String, Unit]("println", lhs)
}

class CodeArray[T](val lhs: Code[Array[T]])(implicit tti: TypeInfo[T]) {
  def apply(i: Code[Int]): Code[T] =
    Code(lhs, i, new InsnNode(tti.aloadOp))

  def update(i: Code[Int], x: Code[T]): Code[Unit] =
    Code(lhs, i, x, new InsnNode(tti.astoreOp))

  def length(): Code[Int] =
    Code(lhs, new InsnNode(ARRAYLENGTH))
}

object Invokeable {
  def apply[T](cls: Class[T], c: Constructor[_]): Invokeable[T, Unit] = new Invokeable[T, Unit](
    cls,
    "<init>",
    isStatic = false,
    isInterface = false,
    INVOKESPECIAL,
    Type.getConstructorDescriptor(c),
    implicitly[ClassTag[Unit]].runtimeClass)

  def apply[T, S](cls: Class[T], m: Method)(implicit sct: ClassTag[S]): Invokeable[T, S] = {
    val isInterface = m.getDeclaringClass.isInterface
    val isStatic = Modifier.isStatic(m.getModifiers)
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

  def lookupConstructor[T](cls: Class[T], parameterTypes: Array[Class[_]]): Invokeable[T, Unit] = {
    val c = cls.getDeclaredConstructor(parameterTypes: _*)
    assert(c != null,
      s"no such method ${ cls.getName }(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    Invokeable(cls, c)
  }
}

class Invokeable[T, S](tcls: Class[T],
  val name: String,
  val isStatic: Boolean,
  val isInterface: Boolean,
  val invokeOp: Int,
  val descriptor: String,
  val concreteReturnType: Class[_])(implicit sct: ClassTag[S]) {
  def invoke(lhs: Code[T], args: Array[Code[_]]): Code[S] =
    new Code[S] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        if (!isStatic && lhs != null)
          lhs.emit(il)
        args.foreach(_.emit(il))
        il += new MethodInsnNode(invokeOp,
          Type.getInternalName(tcls), name, descriptor, isInterface)
        if (concreteReturnType != sct.runtimeClass) {
          // if `m`'s return type is a generic type, we must use an explicit
          // cast to the expected type
          il += new TypeInsnNode(CHECKCAST, Type.getInternalName(sct.runtimeClass))
        }
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

trait Settable[T] {
  def load(): Code[T]

  def store(rhs: Code[T]): Code[Unit]

  def :=(rhs: Code[T]): Code[Unit] = store(rhs)

  def storeAny(rhs: Code[_]): Code[Unit] = store(coerce[T](rhs))
}

class LazyFieldRef[T: TypeInfo](fb: FunctionBuilder[_], name: String, setup: Code[T]) extends Settable[T] {

  private[this] val value: ClassFieldRef[T] = fb.newField[T](name)
  private[this] val present: ClassFieldRef[Boolean] = fb.newField[Boolean](s"${name}_present")

  def load(): Code[T] =
    Code(present.mux(Code._empty[Unit], Code(value := setup, present := true)), value)

  def store(rhs: Code[T]): Code[Unit] =
    throw new UnsupportedOperationException("cannot store new value into LazyFieldRef!")
}

class ClassFieldRef[T: TypeInfo](fb: FunctionBuilder[_], val name: String) extends Settable[T] {
  val desc: String = typeInfo[T].name
  val node: FieldNode = new FieldNode(ACC_PUBLIC, name, desc, null, null)

  fb.cn.fields.asInstanceOf[util.List[FieldNode]].add(node)

  def load(): Code[T] =
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        fb.getArg[java.lang.Object](0).emit(il)
        il += new FieldInsnNode(GETFIELD, fb.name, name, desc)
      }
    }

  def store(rhs: Code[T]): Code[Unit] =
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        fb.getArg[java.lang.Object](0).emit(il)
        rhs.emit(il)
        il += new FieldInsnNode(PUTFIELD, fb.name, name, desc)
      }
    }

  def storeInsn: Code[Unit] = Code(new FieldInsnNode(PUTFIELD, fb.name, name, desc))
}

class LocalRef[T](val i: Int)(implicit tti: TypeInfo[T]) extends Settable[T] {
  def load(): Code[T] =
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        il += new VarInsnNode(tti.loadOp, i)
      }
    }

  def store(rhs: Code[T]): Code[Unit] =
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        rhs.emit(il)
        il += new VarInsnNode(tti.storeOp, i)
      }
    }

  def storeInsn: Code[Unit] = Code(new VarInsnNode(tti.storeOp, i))
}

class LocalRefInt(val v: LocalRef[Int]) extends AnyRef {
  def +=(i: Int): Code[Unit] = {
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        il += new IincInsnNode(v.i, i)
      }
    }
  }

  def ++(): Code[Unit] = +=(1)
}

class FieldRef[T, S](f: Field)(implicit tct: ClassTag[T], sti: TypeInfo[S]) {
  def isStatic: Boolean = Modifier.isStatic(f.getModifiers)

  def getOp = if (isStatic) GETSTATIC else GETFIELD

  def putOp = if (isStatic) PUTSTATIC else PUTFIELD

  def get(): Code[S] = get(Code._empty)

  def get(lhs: Code[T]): Code[S] =
    new Code[S] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        if (!isStatic)
          lhs.emit(il)
        il += new FieldInsnNode(getOp,
          Type.getInternalName(tct.runtimeClass), f.getName, sti.name)
      }
    }

  def put(lhs: Code[T], rhs: Code[S]): Code[Unit] =
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        if (!isStatic)
          lhs.emit(il)
        rhs.emit(il)
        il += new FieldInsnNode(putOp,
          Type.getInternalName(tct.runtimeClass), f.getName, sti.name)
      }
    }
}

class CodeObject[T <: AnyRef : ClassTag](val lhs: Code[T]) {
  def get[S](field: String)(implicit sct: ClassTag[S], sti: TypeInfo[S]): Code[S] =
    FieldRef[T, S](field).get(lhs)

  def put[S](field: String, rhs: Code[S])(implicit sct: ClassTag[S], sti: TypeInfo[S]): Code[Unit] =
    FieldRef[T, S](field).put(lhs, rhs)

  def invoke[S](method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]])
    (implicit sct: ClassTag[S]): Code[S] =
    Invokeable.lookupMethod[T, S](implicitly[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]], method, parameterTypes).invoke(lhs, args)

  def invoke[S](method: String)(implicit sct: ClassTag[S]): Code[S] =
    invoke[S](method, Array[Class[_]](), Array[Code[_]]())

  def invoke[A1, S](method: String, a1: Code[A1])(implicit a1ct: ClassTag[A1],
    sct: ClassTag[S]): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def invoke[A1, A2, S](method: String, a1: Code[A1], a2: Code[A2])(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2],
    sct: ClassTag[S]): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def invoke[A1, A2, A3, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3])
    (implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], sct: ClassTag[S]): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3))

  def invoke[A1, A2, A3, A4, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4])
    (implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], sct: ClassTag[S]): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4))

  def invoke[A1, A2, A3, A4, A5, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4], a5: Code[A5])
    (implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4], a5ct: ClassTag[A5], sct: ClassTag[S]): Code[S] =
    invoke[S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass, a5ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4, a5))
}

class CodeNullable[T >: Null : TypeInfo](val lhs: Code[T]) {
  def isNull: CodeConditional = new CodeConditional {
      def emitConditional(il: Growable[AbstractInsnNode], ltrue: LabelNode, lfalse: LabelNode): Unit = {
        lhs.emit(il)
        il += new JumpInsnNode(IFNULL, ltrue)
        il += new JumpInsnNode(GOTO, lfalse)
      }
    }

  def ifNull[T](cnullcase: Code[T], cnonnullcase: Code[T]): Code[T] =
    isNull.mux(cnullcase, cnonnullcase)

  def mapNull[U >: Null](cnonnullcase: Code[U]): Code[U] =
    ifNull[U](Code._null[U], cnonnullcase)

  // Ideally this would not use a local variable, but I need a richer way to
  // talk about the way `Code` modifies the stack.
  def mapNull[U >: Null](cnonnullcase: Code[T] => Code[U]): CM[Code[U]] = for (
    (stx, x) <- CM.memoize(lhs)
  ) yield Code(stx,
    x.ifNull[U](Code._null[U], cnonnullcase(x)))

  def mapNullM[U >: Null](cnonnullcase: Code[T] => CM[Code[U]]): CM[Code[U]] = for (
    (stx, x) <- CM.memoize(lhs);
    result <- cnonnullcase(x)
  ) yield Code(stx,
    x.ifNull[U](Code._null[U], result))
}
