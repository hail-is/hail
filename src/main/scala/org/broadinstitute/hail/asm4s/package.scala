package org.broadinstitute.hail

import java.lang.reflect.Method

import org.objectweb.asm.Type
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.language.implicitConversions
import scala.reflect.ClassTag

package object asm4s {

  trait TypeInfo[T] {
    val name: String
    val loadOp: Int
    val storeOp: Int
    val aloadOp: Int
    val astoreOp: Int
    val returnOp: Int
    val slots: Int = 1

    def newArray(): AbstractInsnNode
  }

  implicit object BooealnInfo extends TypeInfo[Boolean] {
    val name = "Z"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN
    val newarrayOp = NEWARRAY

    def newArray() = new IntInsnNode(NEWARRAY, T_BOOLEAN)
  }

  implicit object ByteInfo extends TypeInfo[Byte] {
    val name = "B"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN
    val newarrayOp = NEWARRAY

    def newArray() = new IntInsnNode(NEWARRAY, T_BYTE)
  }

  implicit object ShortInfo extends TypeInfo[Short] {
    val name = "S"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN
    val newarrayOp = NEWARRAY

    def newArray() = new IntInsnNode(NEWARRAY, T_SHORT)
  }

  implicit object IntInfo extends TypeInfo[Int] {
    val name = "I"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN

    def newArray() = new IntInsnNode(NEWARRAY, T_INT)
  }

  implicit object LongInfo extends TypeInfo[Long] {
    val name = "J"
    val loadOp = LLOAD
    val storeOp = LSTORE
    val aloadOp = LALOAD
    val astoreOp = LASTORE
    val returnOp = LRETURN
    override val slots = 2

    def newArray() = new IntInsnNode(NEWARRAY, T_LONG)
  }

  implicit object FloatInfo extends TypeInfo[Float] {
    val name = "F"
    val loadOp = FLOAD
    val storeOp = FSTORE
    val aloadOp = FALOAD
    val astoreOp = FASTORE
    val returnOp = FRETURN

    def newArray() = new IntInsnNode(NEWARRAY, T_FLOAT)
  }

  implicit object DoubleInfo extends TypeInfo[Double] {
    val name = "D"
    val loadOp = DLOAD
    val storeOp = DSTORE
    val aloadOp = DALOAD
    val astoreOp = DASTORE
    val returnOp = DRETURN
    override val slots = 2

    def newArray() = new IntInsnNode(NEWARRAY, T_DOUBLE)
  }

  implicit object CharInfo extends TypeInfo[Char] {
    val name = "C"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN
    override val slots = 2

    def newArray() = new IntInsnNode(NEWARRAY, T_CHAR)
  }

  implicit def classInfo[C <: AnyRef](implicit cct: ClassTag[C]): TypeInfo[C] = {
    new TypeInfo[C] {
      val name = Type.getDescriptor(cct.runtimeClass)
      val loadOp = ALOAD
      val storeOp = ASTORE
      val aloadOp = AALOAD
      val astoreOp = AASTORE
      val returnOp = ARETURN

      def newArray() = new TypeInsnNode(ANEWARRAY, name)
    }
  }

  def loadClass(className: String, b: Array[Byte]): Class[_] = {
    // override classDefine (as it is protected) and define the class.
    var clazz: Class[_] = null
    val loader: ClassLoader = ClassLoader.getSystemClassLoader
    val cls: Class[_] = Class.forName("java.lang.ClassLoader")
    val method: Method = cls.getDeclaredMethod("defineClass", classOf[String], classOf[Array[Byte]], classOf[Int], classOf[Int])

    // protected method invocaton
    method.setAccessible(true)

    clazz = method.invoke(loader, null, b, new Integer(0), new Integer(b.length)).asInstanceOf[Class[_]]

    method.setAccessible(false)

    clazz
  }

  def ??? = throw new UnsupportedOperationException

  implicit def toCodeBoolean(c: Code[Boolean]): CodeBoolean = new CodeBoolean(c)

  implicit def toCodeInt(c: Code[Int]): CodeInt = new CodeInt(c)

  implicit def toCodeDouble(c: Code[Double]): CodeDouble = new CodeDouble(c)

  implicit def toCodeArray[T](c: Code[Array[T]])(implicit tti: TypeInfo[T]): CodeArray[T] = new CodeArray(c)

  implicit def toCodeObject[T <: AnyRef](c: Code[T])(implicit tti: TypeInfo[T], tct: ClassTag[T]): CodeObject[T] =
    new CodeObject(c)

  implicit def toCode[T](insn: => AbstractInsnNode): Code[T] = new Code[T] {
    def emit(il: InsnList): Unit = {
      il.add(insn)
    }
  }

  implicit def toCode[T](f: LocalRef[T]): Code[T] = f.load()

  implicit def toCodeInt(f: LocalRef[Int]): CodeInt = new CodeInt(f.load())

  implicit def toCodeDouble(f: LocalRef[Double]): CodeDouble = new CodeDouble(f.load())

  implicit def toCodeBoolean(f: LocalRef[Boolean]): CodeBoolean = new CodeBoolean(f.load())

  implicit def toCodeObject[T <: AnyRef](f: LocalRef[T])(implicit tti: TypeInfo[T], tct: ClassTag[T]): CodeObject[T] = new CodeObject[T](f.load())

  implicit def const(b: Boolean): Code[Boolean] = Code(new LdcInsnNode(if (b) 1 else 0))

  implicit def const(i: Int): Code[Int] = Code(new LdcInsnNode(i))

  implicit def const(d: Double): Code[Double] = Code(new LdcInsnNode(d))
}
