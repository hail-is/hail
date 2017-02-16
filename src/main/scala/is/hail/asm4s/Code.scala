package is.hail.asm4s

import java.lang.reflect.{Constructor, Field, Method, Modifier}

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.reflect.ClassTag

object Code {
  def apply[T](insn: => AbstractInsnNode): Code[T] = new Code[T] {
    def emit(il: InsnList): Unit = {
      il.add(insn)
    }
  }

  def apply[S1, S2](c1: Code[S1], c2: Code[S2]): Code[S2] =
    new Code[S2] {
      def emit(il: InsnList): Unit = {
        c1.emit(il)
        c2.emit(il)
      }
    }

  def apply[S1, S2, S3](c1: Code[S1], c2: Code[S2], c3: Code[S3]): Code[S3] =
    new Code[S3] {
      def emit(il: InsnList): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
      }
    }

  def apply[S1, S2, S3, S4](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4]): Code[S4] =
    new Code[S4] {
      def emit(il: InsnList): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5]): Code[S5] =
    new Code[S5] {
      def emit(il: InsnList): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
      }
    }

  def apply[S1, S2, S3, S4, S5, S6](c1: Code[S1], c2: Code[S2], c3: Code[S3], c4: Code[S4], c5: Code[S5], c6: Code[S6]): Code[S6] =
    new Code[S6] {
      def emit(il: InsnList): Unit = {
        c1.emit(il)
        c2.emit(il)
        c3.emit(il)
        c4.emit(il)
        c5.emit(il)
        c6.emit(il)
      }
    }
}

trait Code[T] {
  self =>
  def emit(il: InsnList): Unit

  def empty: Code[Unit] = new Code[Unit] {
    def emit(il: InsnList): Unit = {
      // nothing
    }
  }

  def compare(opcode: Int, rhs: Code[T]): CodeConditional =
    new CodeConditional {
      def emitConditional(il: InsnList): (LabelNode, LabelNode) = {
        val ltrue = new LabelNode
        val lfalse = new LabelNode
        self.emit(il)
        rhs.emit(il)
        il.add(new JumpInsnNode(opcode, ltrue))
        il.add(new JumpInsnNode(GOTO, lfalse))
        (ltrue, lfalse)
      }
    }
}

trait CodeConditional extends Code[Boolean] { self =>
  def emit(il: InsnList): Unit = {
    val lafter = new LabelNode
    val (ltrue, lfalse) = emitConditional(il)
    il.add(lfalse)
    il.add(new LdcInsnNode(0))
    il.add(new JumpInsnNode(GOTO, lafter))
    il.add(ltrue)
    il.add(new LdcInsnNode(1))
    il.add(lafter)
  }

  // returns (ltrue, lfalse)
  def emitConditional(il: InsnList): (LabelNode, LabelNode)

  def unary_!(): CodeConditional =
    new CodeConditional {
      def emitConditional(il: InsnList): (LabelNode, LabelNode) = {
        val (ltrue, lfalse) = self.emitConditional(il)
        (lfalse, ltrue)
      }
    }
}

class CodeBoolean(val lhs: Code[Boolean]) extends AnyVal {
  def toConditional: CodeConditional = lhs match {
    case cond: CodeConditional =>
      cond

    case _ =>
      new CodeConditional {
        def emitConditional(il: InsnList): (LabelNode, LabelNode) = {
          val ltrue = new LabelNode
          val lfalse = new LabelNode
          lhs.emit(il)
          il.add(new JumpInsnNode(IFEQ, lfalse))
          il.add(new JumpInsnNode(GOTO, ltrue))
          (ltrue, lfalse)
        }
      }
  }

  def unary_!(): Code[Boolean] =
    !lhs.toConditional

  def mux[T](cthen: Code[T], celse: Code[T]): Code[T] = {
    val cond = lhs.toConditional
    new Code[T] {
      def emit(il: InsnList): Unit = {
        val lafter = new LabelNode
        val (ltrue, lfalse) = cond.emitConditional(il)
        il.add(lfalse)
        celse.emit(il)
        il.add(new JumpInsnNode(GOTO, lafter))
        il.add(ltrue)
        cthen.emit(il)
        // fall through
        il.add(lafter)
      }
    }
  }
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

  def ceq(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPEQ, rhs)

  def cne(rhs: Code[Int]): Code[Boolean] = lhs.compare(IF_ICMPNE, rhs)
}

class CodeDouble(val lhs: Code[Double]) extends AnyVal {
  def +(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DADD))

  def -(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DSUB))

  def *(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DMUL))

  def /(rhs: Code[Double]): Code[Double] = Code(lhs, rhs, new InsnNode(DDIV))

  // FIXME DCMPG vs DCMPL
  def compare(rhs: Code[Double]): Code[Int] = Code(lhs, rhs, new InsnNode(DCMPG))

  def >(rhs: Code[Double]): Code[Boolean] = compare(rhs) > 0

  def >=(rhs: Code[Double]): Code[Boolean] = compare(rhs) >= 0

  def <(rhs: Code[Double]): Code[Boolean] = compare(rhs) < 0

  def <=(rhs: Code[Double]): Code[Boolean] = compare(rhs) <= 0

  def ceq(rhs: Code[Double]): Code[Boolean] = compare(rhs).ceq(0)

  def cne(rhs: Code[Double]): Code[Boolean] = compare(rhs).cne(0)
}

class CodeArray[T](val lhs: Code[Array[T]])(implicit tti: TypeInfo[T]) {
  def apply(i: Code[Int]): Code[T] =
    Code(lhs, i, new InsnNode(tti.aloadOp))

  def update(i: Code[Int], x: Code[T]): Code[Unit] =
    Code(lhs, i, x, new InsnNode(tti.astoreOp))
}

object Invokeable {
  def apply[T](c: Constructor[_])(implicit tct: ClassTag[T]): Invokeable[T, Unit] = new Invokeable[T, Unit]("<init>",
    isStatic = false,
    isInterface = false,
    INVOKESPECIAL,
    Type.getConstructorDescriptor(c))

  def apply[T, S](m: Method)(implicit tct: ClassTag[T]): Invokeable[T, S] = {
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
      Type.getMethodDescriptor(m))
  }

  def lookupMethod[T, S](method: String, parameterTypes: Array[Class[_]])(implicit tct: ClassTag[T], sct: ClassTag[S]): Invokeable[T, S] = {
    val m = tct.runtimeClass.getDeclaredMethod(method, parameterTypes: _*)
    assert(m != null,
      s"no such method ${tct.runtimeClass.getName}.$method(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    assert(m.getReturnType == sct.runtimeClass,
      s"when invoking ${tct.runtimeClass.getName}.$method(): ${m.getReturnType.getName}: wrong return type ${sct.runtimeClass.getName}")

    Invokeable(m)
  }

  def lookupConstructor[T](parameterTypes: Array[Class[_]])(implicit tct: ClassTag[T]): Invokeable[T, Unit] = {
    val c = tct.runtimeClass.getDeclaredConstructor(parameterTypes: _*)
    assert(c != null,
      s"no such method ${tct.runtimeClass.getName}(${
        parameterTypes.map(_.getName).mkString(", ")
      })")

    Invokeable[T](c)
  }
}

class Invokeable[T, S](val name: String,
                       val isStatic: Boolean,
                       val isInterface: Boolean,
                       val invokeOp: Int,
                       val descriptor: String)(implicit tct: ClassTag[T]) {
  def invoke(lhs: Code[T], args: Array[Code[_]]): Code[S] =
    new Code[S] {
      def emit(il: InsnList): Unit = {
        if (!isStatic && lhs != null)
          lhs.emit(il)
        args.foreach(_.emit(il))
        il.add(new MethodInsnNode(invokeOp,
          Type.getInternalName(tct.runtimeClass), name, descriptor, isInterface))
      }
    }
}

object FieldRef {
  def apply[T, S](field: String)(implicit tct: ClassTag[T], sct: ClassTag[S], sti: TypeInfo[S]): FieldRef[T, S] = {
    val f = tct.runtimeClass.getDeclaredField(field)
    assert(f.getType == sct.runtimeClass,
      s"when getting field ${tct.runtimeClass.getName}.$field: ${f.getType.getName}: wrong type ${sct.runtimeClass.getName} ")

    new FieldRef(f)
  }
}

class LocalRef[T](i: Int)(implicit tti: TypeInfo[T]) {
  def load(): Code[T] =
    new Code[T] {
      def emit(il: InsnList): Unit = {
        il.add(new IntInsnNode(tti.loadOp, i))
      }
    }

  def store(rhs: Code[T]): Code[T] =
    new Code[T] {
      def emit(il: InsnList): Unit = {
        rhs.emit(il)
        il.add(new IntInsnNode(tti.storeOp, i))
      }
    }
}

class FieldRef[T, S](f: Field)(implicit tct: ClassTag[T], sti: TypeInfo[S]) {
  def isStatic: Boolean = Modifier.isStatic(f.getModifiers)

  def getOp = if (isStatic) GETSTATIC else GETFIELD

  def putOp = if (isStatic) PUTSTATIC else PUTFIELD

  def get(lhs: Code[T]): Code[S] =
    new Code[S] {
      def emit(il: InsnList): Unit = {
        if (!isStatic)
          lhs.emit(il)
        il.add(new FieldInsnNode(getOp,
          Type.getInternalName(tct.runtimeClass), f.getName, sti.name))
      }
    }

  def put(lhs: Code[T], rhs: Code[S]): Code[Unit] =
    new Code[Unit] {
      def emit(il: InsnList): Unit = {
        if (!isStatic)
          lhs.emit(il)
        rhs.emit(il)
        il.add(new FieldInsnNode(putOp,
          Type.getInternalName(tct.runtimeClass), f.getName, sti.name))
      }
    }
}

class CodeObject[T <: AnyRef](val lhs: Code[T])(implicit tct: ClassTag[T], tti: TypeInfo[T]) {
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
}
