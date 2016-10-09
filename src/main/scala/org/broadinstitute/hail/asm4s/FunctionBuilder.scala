package org.broadinstitute.hail.asm4s

import java.io.PrintWriter

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._
import java.util

import org.objectweb.asm.util.{Textifier, TraceClassVisitor}
import org.objectweb.asm.{ClassReader, ClassWriter, Type}

import scala.language.implicitConversions
import scala.reflect.ClassTag

object FunctionBuilder {
  var count = 0

  def newUniqueID(): Int = {
    val id = count
    count += 1
    id
  }
}

abstract class FunctionBuilder[R](parameterTypeInfo: Array[TypeInfo[_]], returnTypeInfo: TypeInfo[R],
                                  packageName: String = "is/hail/codegen/generated") {

  import FunctionBuilder._

  val cn = new ClassNode()
  cn.version = V1_8
  cn.access = ACC_PUBLIC

  cn.name = packageName + "/C" + newUniqueID()
  cn.superName = "java/lang/Object"

  def signature: String = s"(${parameterTypeInfo.map(_.name).mkString})${returnTypeInfo.name}"

  val mn = new MethodNode(ACC_PUBLIC + ACC_STATIC, "f", signature, null, null)
  // FIXME why is cast necessary?
  cn.methods.asInstanceOf[util.List[MethodNode]].add(mn)
  val il = mn.instructions

  val start = new LabelNode
  val end = new LabelNode

  val layout: Array[Int] =
    parameterTypeInfo.scanLeft(0) { case (prev, ti) => prev + ti.slots }
  val argIndex: Array[Int] = layout.init
  var locals: Int = layout.last

  def allocLocal[T]()(implicit tti: TypeInfo[T]): Int = {
    val i = locals
    locals += tti.slots

    mn.localVariables.asInstanceOf[util.List[LocalVariableNode]]
      .add(new LocalVariableNode("local" + i, tti.name, null, start, end, i))
    i
  }

  def newLocal[T]()(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef[T](allocLocal[T]())

  def invokeStatic[T, S](method: String, parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit tct: ClassTag[T], sct: ClassTag[S]): Code[S] = {
    val m = Invokeable.lookupMethod[T, S](method, parameterTypes)
    assert(m.isStatic)
    m.invoke(null, args)
  }

  def invokeStatic[T, S](method: String)(implicit tct: ClassTag[T], sct: ClassTag[S]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](), Array[Code[_]]())

  def invokeStatic[T, A1, S](method: String, a1: Code[A1])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def invokeStatic[T, A1, A2, S](method: String, a1: Code[A1], a2: Code[A2])(implicit tct: ClassTag[T], sct: ClassTag[S], a1ct: ClassTag[A1], a2ct: ClassTag[A2]): Code[S] =
    invokeStatic[T, S](method, Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def getStatic[T, S](field: String)(implicit tct: ClassTag[T], sct: ClassTag[S], sti: TypeInfo[S]): Code[S] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.get(null)
  }

  def putStatic[T, S](field: String, rhs: Code[S])(implicit tct: ClassTag[T], sct: ClassTag[S], sti: TypeInfo[S]): Code[Unit] = {
    val f = FieldRef[T, S](field)
    assert(f.isStatic)
    f.put(null, rhs)
  }

  def newArray[T](size: Code[Int])(implicit tti: TypeInfo[T],
                                   atti: TypeInfo[Array[T]]): (Code[Unit], Code[Array[T]]) = {
    val arri = allocLocal[Array[T]]()
    (new Code[Unit] {
      def emit(il: InsnList): Unit = {
        size.emit(il)
        il.add(tti.newArray())
        il.add(new IntInsnNode(atti.storeOp, arri))
      }
    }, new Code[Array[T]] {
      def emit(il: InsnList): Unit = {
        il.add(new IntInsnNode(atti.loadOp, arri))
      }
    })
  }

  def newInstance[T](parameterTypes: Array[Class[_]], args: Array[Code[_]])(implicit tct: ClassTag[T], tti: TypeInfo[T]): (Code[Unit], Code[T]) = {
    val inst = allocLocal[T]()
    val ctor = Invokeable.lookupConstructor[T](Array()).invoke(null, Array())
    (new Code[Unit] {
      def emit(il: InsnList): Unit = {
        il.add(new TypeInsnNode(NEW, Type.getInternalName(tct.runtimeClass)))
        il.add(new InsnNode(DUP))
        il.add(new IntInsnNode(tti.storeOp, inst))
        ctor.emit(il)
      }
    },
      new Code[T] {
        def emit(il: InsnList): Unit = {
          il.add(new IntInsnNode(tti.loadOp, inst))
        }
      })
  }

  def newInstance[T]()(implicit tct: ClassTag[T], tti: TypeInfo[T]): (Code[Unit], Code[T]) =
    newInstance[T](Array[Class[_]](), Array[Code[_]]())

  def newInstance[T, A1](a1: Code[A1])(implicit a1ct: ClassTag[A1],
                                       tct: ClassTag[T], tti: TypeInfo[T]): (Code[Unit], Code[T]) =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass), Array[Code[_]](a1))

  def newInstance[T, A1, A2](a1: Code[A1], a2: Code[A2])(implicit a1ct: ClassTag[A1], a2ct: ClassTag[A2],
                                                         tct: ClassTag[T], tti: TypeInfo[T]): (Code[Unit], Code[T]) =
    newInstance[T](Array[Class[_]](a1ct.runtimeClass, a2ct.runtimeClass), Array[Code[_]](a1, a2))

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): LocalRef[T] = {
    assert(i >= 0)
    assert(i < parameterTypeInfo.length)
    new LocalRef[T](argIndex(i))
  }

  def resultClass(c: Code[R]): Class[_] = {
    mn.instructions.add(start)
    c.emit(mn.instructions)
    mn.instructions.add(new InsnNode(returnTypeInfo.returnOp))
    mn.instructions.add(end)

    val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS +
      ClassWriter.COMPUTE_FRAMES)
    cn.accept(cw)
    val b = cw.toByteArray
    val clazz = loadClass(null, b)

    // print bytecode for debugging
    /*
    val cr = new ClassReader(b)
    val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(System.out))
    cr.accept(tcv, 0)
    */

    clazz
  }

  def whileLoop(condition: Code[Boolean], body: Code[_]): Code[Unit] = {
    val l1 = new LabelNode
    val l2 = new LabelNode
    new Code[Unit] {
      def emit(il: InsnList): Unit = {
        il.add(l1)
        condition.emit(il)
        il.add(new LdcInsnNode(0))
        il.add(new JumpInsnNode(IF_ICMPEQ, l2))
        body.emit(il)
        il.add(new JumpInsnNode(GOTO, l1))
        il.add(l2)
      }
    }
  }
}

class Function0Builder[R](implicit rti: TypeInfo[R]) extends FunctionBuilder[R](Array[TypeInfo[_]](), rti) {
  def result(c: Code[R]): () => R = {
    val clazz = resultClass(c)
    val m = clazz.getMethod("f")
    () => m.invoke(null).asInstanceOf[R]
  }
}

class Function1Builder[A, R](implicit act: ClassTag[A], ati: TypeInfo[A],
                             rti: TypeInfo[R]) extends FunctionBuilder[R](Array[TypeInfo[_]](ati), rti) {
  def arg1 = getArg[A](0)

  def result(c: Code[R]): (A) => R = {
    val clazz = resultClass(c)
    val m = clazz.getMethod("f", act.runtimeClass)
    (a: A) => m.invoke(null, a.asInstanceOf[AnyRef]).asInstanceOf[R]
  }
}

class Function2Builder[A1, A2, R](implicit a1ct: ClassTag[A1], a1ti: TypeInfo[A1],
                                  a2ct: ClassTag[A2], a2ti: TypeInfo[A2],
                                  rti: TypeInfo[R]) extends FunctionBuilder[R](Array[TypeInfo[_]](a1ti, a2ti), rti) {
  def arg1 = getArg[A1](0)

  def arg2 = getArg[A2](1)

  def result(c: Code[R]): (A1, A2) => R = {
    val clazz = resultClass(c)
    val m = clazz.getMethod("f", a1ct.runtimeClass, a2ct.runtimeClass)
    (a1: A1, a2: A2) => m.invoke(null, a1.asInstanceOf[AnyRef], a2.asInstanceOf[AnyRef]).asInstanceOf[R]
  }
}