package is.hail.asm4s

import is.hail.expr.CM
import java.lang.reflect.{Constructor, Field, Method, Modifier}

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
        Invokeable.lookupConstructor[T](parameterTypes).invoke(null, args).emit(il)
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

  def invokeStatic[T, S](method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit tct: ClassTag[T], sct: ClassTag[S]): Code[S] = {
    val m = Invokeable.lookupMethod[T, S](method, parameterTypes)
    assert(m.isStatic)
    m.invoke(null, args)
  }

  def invokeStatic[T, S](method: String)(implicit tct: ClassTag[T], sct: ClassTag[S]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](), Array[Code[_]]())

  def invokeStatic[T, A1, S](method: String, a1: Code[A1])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))(tct, sct)

  def invokeStatic[T, A1, A2, S](method: String, a1: Code[A1], a2: Code[A2])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))(tct, sct)

  def invokeStatic[T, A1, A2, A3, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass), Array[Code[_]](a1, a2, a3))(tct, sct)

  def invokeStatic[T, A1, A2, A3, A4, S](method: String, a1: Code[A1], a2: Code[A2], a3: Code[A3], a4: Code[A4])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2], a3ct: ClassTag[A3], a4ct: ClassTag[A4]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass, a3ct.runtimeClass, a4ct.runtimeClass), Array[Code[_]](a1, a2, a3, a4))(tct, sct)

  def _null[T >: Null]: Code[T] = Code(new InsnNode(ACONST_NULL))

  // FIXME: code should really carry around the stack so this type can be correct
  // Currently, this is a huge potential place for errors.
  def _pop[T]: Code[T] = Code(new InsnNode(POP))

  // FIXME: code should really carry around the stack so this type can be correct
  // Currently, this is a huge potential place for errors.
  def _empty[T]: Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
    }
  }

  def _throw[T <: java.lang.Throwable, U](cerr: Code[T]): Code[U] = Code(cerr, new InsnNode(ATHROW))

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
}

class CodeInt(val lhs: Code[Int]) extends AnyVal {
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

  def ^(rhs: Code[Int]): Code[Int] = Code(lhs, rhs, new InsnNode(IXOR))

  def unary_~(): Code[Int] = lhs ^ const(-1)

  def ceq(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPEQ, rhs)

  def cne(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPNE, rhs)

  def negate(): Code[Int] = Code(lhs, new InsnNode(INEG))

  def toI: Code[Int] = lhs

  def toL: Code[Long] = Code(lhs, new InsnNode(I2L))

  def toF: Code[Float] = Code(lhs, new InsnNode(I2F))

  def toD: Code[Double] = Code(lhs, new InsnNode(I2D))

  def toB: Code[Byte] = Code(lhs, new InsnNode(I2B))
}

class CodeLong(val lhs: Code[Long]) extends AnyVal {
  def +(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LADD))

  def -(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LSUB))

  def *(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LMUL))

  def /(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LDIV))

  def compare(rhs: Code[Long]): Code[Int] = Code(lhs, rhs, new InsnNode(LCMP))

  def <(rhs: Code[Long]): Code[Boolean] = compare(rhs) < 0

  def >(rhs: Code[Long]): Code[Boolean] = compare(rhs) > 0

  def ceq(rhs: Code[Long]): Code[Boolean] = compare(rhs) ceq 0

  def cne(rhs: Code[Long]): Code[Boolean] = compare(rhs) cne 0

  def >>(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LSHR))

  def <<(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LSHL))

  def >>>(rhs: Code[Int]): Code[Long] = Code(lhs, rhs, new InsnNode(LUSHR))

  def &(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LAND))

  def ^(rhs: Code[Long]): Code[Long] = Code(lhs, rhs, new InsnNode(LXOR))

  def unary_~(): Code[Long] = lhs ^ const(-1L)

  def toI: Code[Int] = Code(lhs, new InsnNode(L2I))

  def toL: Code[Long] = lhs

  def toF: Code[Float] = Code(lhs, new InsnNode(L2F))

  def toD: Code[Double] = Code(lhs, new InsnNode(L2D))
}

class CodeFloat(val lhs: Code[Float]) extends AnyVal {
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
}

class CodeDouble(val lhs: Code[Double]) extends AnyVal {
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
  def apply[T](c: Constructor[_])(implicit tct: ClassTag[T]): Invokeable[T, Unit] = new Invokeable[T, Unit]("<init>",
    isStatic = false,
    isInterface = false,
    INVOKESPECIAL,
    Type.getConstructorDescriptor(c),
    implicitly[ClassTag[Unit]].runtimeClass)

  def apply[T, S](m: Method)(implicit tct: ClassTag[T], sct: ClassTag[S]): Invokeable[T, S] = {
    val isInterface = m.getDeclaringClass.isInterface
    val isStatic = Modifier.isStatic(m.getModifiers)
    assert(!(isInterface && isStatic))
    new Invokeable[T, S](m.getName,
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

  def lookupMethod[T, S](method: String, parameterTypes: Array[Class[_]])(implicit tct: ClassTag[T], sct: ClassTag[S]): Invokeable[T, S] = {
    val m = tct.runtimeClass.getMethod(method, parameterTypes: _*)
    assert(m != null,
      s"no such method ${ tct.runtimeClass.getName }.$method(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    // generic type parameters return java.lang.Object instead of the correct class
    assert(m.getReturnType.isAssignableFrom(sct.runtimeClass),
      s"when invoking ${ tct.runtimeClass.getName }.$method(): ${ m.getReturnType.getName }: wrong return type ${ sct.runtimeClass.getName }")

    Invokeable(m)
  }

  def lookupConstructor[T](parameterTypes: Array[Class[_]])(implicit tct: ClassTag[T]): Invokeable[T, Unit] = {
    val c = tct.runtimeClass.getDeclaredConstructor(parameterTypes: _*)
    assert(c != null,
      s"no such method ${ tct.runtimeClass.getName }(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    Invokeable[T](c)
  }
}

class Invokeable[T, S](val name: String,
  val isStatic: Boolean,
  val isInterface: Boolean,
  val invokeOp: Int,
  val descriptor: String,
  val concreteReturnType: Class[_])(implicit tct: ClassTag[T], sct: ClassTag[S]) {
  def invoke(lhs: Code[T], args: Array[Code[_]]): Code[S] =
    new Code[S] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        if (!isStatic && lhs != null)
          lhs.emit(il)
        args.foreach(_.emit(il))
        il += new MethodInsnNode(invokeOp,
          Type.getInternalName(tct.runtimeClass), name, descriptor, isInterface)
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

class LocalRef[T](val i: Int)(implicit tti: TypeInfo[T]) {
  def load(): Code[T] =
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        il += new IntInsnNode(tti.loadOp, i)
      }
    }

  def store(rhs: Code[T]): Code[Unit] =
    new Code[Unit] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        rhs.emit(il)
        il += new IntInsnNode(tti.storeOp, i)
      }
    }

  def storeInsn: Code[Unit] = Code(new IntInsnNode(tti.storeOp, i))

  def :=(rhs: Code[T]): Code[Unit] = store(rhs)
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

class CodeObject[T >: Null](val lhs: Code[T])(implicit tct: ClassTag[T], tti: TypeInfo[T]) {
  def get[S](field: String)(implicit sct: ClassTag[S], sti: TypeInfo[S]): Code[S] =
    FieldRef[T, S](field).get(lhs)

  def put[S](field: String, rhs: Code[S])(implicit sct: ClassTag[S], sti: TypeInfo[S]): Code[Unit] =
    FieldRef[T, S](field).put(lhs, rhs)

  def invoke[S](method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]])
    (implicit sct: ClassTag[S]): Code[S] =
    Invokeable.lookupMethod[T, S](method, parameterTypes).invoke(lhs, args)

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

  def ifNull[T](cnullcase: Code[T], cnonnullcase: Code[T]): Code[T] =
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        val lnull = new LabelNode
        val lafter = new LabelNode
        lhs.emit(il)
        il += new JumpInsnNode(IFNULL, lnull)
        cnonnullcase.emit(il)
        il += new JumpInsnNode(GOTO, lafter)
        il += lnull
        cnullcase.emit(il)
        // fall through
        il += lafter
      }
    }

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
