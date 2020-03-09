package is.hail

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.language.implicitConversions
import scala.reflect.ClassTag

package asm4s {
  trait TypeInfo[T] {
    val name: String
    val iname: String = name
    def loadOp: Int
    def storeOp: Int
    def aloadOp: Int
    def astoreOp: Int
    val returnOp: Int
    def slots: Int = 1

    def newArray(): AbstractInsnNode
  }

  class ClassInfo[C <: AnyRef](implicit val cct: ClassTag[C]) extends TypeInfo[C] {
    val name = Type.getDescriptor(cct.runtimeClass)
    override val iname = Type.getInternalName(cct.runtimeClass)
    val loadOp = ALOAD
    val storeOp = ASTORE
    val aloadOp = AALOAD
    val astoreOp = AASTORE
    val returnOp = ARETURN

    def newArray() = new TypeInsnNode(ANEWARRAY, iname)
  }

  class ArrayInfo[T](implicit val tct: ClassTag[Array[T]]) extends TypeInfo[Array[T]] {
    val name = Type.getDescriptor(tct.runtimeClass)
    override val iname = Type.getInternalName(tct.runtimeClass)
    val loadOp = ALOAD
    val storeOp = ASTORE
    val aloadOp = AALOAD
    val astoreOp = AASTORE
    val returnOp = ARETURN

    def newArray() = new TypeInsnNode(ANEWARRAY, iname)
  }
}

package object asm4s {

  def typeInfo[T](implicit tti: TypeInfo[T]): TypeInfo[T] = tti

  def coerce[T](c: Code[_]): Code[T] =
    c.asInstanceOf[Code[T]]

  def coerce[T](c: Value[_]): Value[T] =
    c.asInstanceOf[Value[T]]

  def coerce[T](c: Settable[_]): Settable[T] =
    c.asInstanceOf[Settable[T]]

  implicit object BooleanInfo extends TypeInfo[Boolean] {
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
    val aloadOp = BALOAD
    val astoreOp = BASTORE
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

  implicit object UnitInfo extends TypeInfo[Unit] {
    val name = "V"
    def loadOp = ???
    def storeOp = ???
    def aloadOp = ???
    def astoreOp = ???
    val returnOp = RETURN
    override def slots = ???

    def newArray() = ???
  }

  implicit def classInfo[C <: AnyRef](implicit cct: ClassTag[C]): TypeInfo[C] =
    new ClassInfo

  implicit def arrayInfo[T](implicit cct: ClassTag[Array[T]]): TypeInfo[Array[T]] =
    new ArrayInfo

  object HailClassLoader extends ClassLoader(getClass.getClassLoader) {
    def loadOrDefineClass(name: String, b: Array[Byte]): Class[_] = {
      getClassLoadingLock(name).synchronized {
        try {
          loadClass(name)
        } catch {
          case e: java.lang.ClassNotFoundException =>
            defineClass(name, b, 0, b.length)
        }
      }
    }
  }

  def loadClass(className: String, b: Array[Byte]): Class[_] =
    HailClassLoader.loadOrDefineClass(className, b)

  def loadClass(className: String): Class[_] =
    HailClassLoader.loadClass(className)

  def ??? = throw new UnsupportedOperationException

  implicit def toCodeBoolean(c: Code[Boolean]): CodeBoolean = new CodeBoolean(c)

  implicit def toCodeInt(c: Code[Int]): CodeInt = new CodeInt(c)

  implicit def byteToCodeInt(c: Code[Byte]): Code[Int] = new Code[Int] {
    def emit(il: Growable[AbstractInsnNode]) = c.emit(il)
  }

  implicit def byteToCodeInt2(c: Code[Byte]): CodeInt = toCodeInt(byteToCodeInt(c))

  implicit def toCodeLong(c: Code[Long]): CodeLong = new CodeLong(c)

  implicit def toCodeFloat(c: Code[Float]): CodeFloat = new CodeFloat(c)

  implicit def toCodeDouble(c: Code[Double]): CodeDouble = new CodeDouble(c)

  implicit def toCodeChar(c: Code[Char]): CodeChar = new CodeChar(c)

  implicit def toCodeString(c: Code[String]): CodeString = new CodeString(c)

  implicit def toCodeArray[T](c: Code[Array[T]])(implicit tti: TypeInfo[T]): CodeArray[T] = new CodeArray(c)

  implicit def toCodeObject[T <: AnyRef : ClassTag](c: Code[T]): CodeObject[T] =
    new CodeObject(c)

  implicit def toCodeNullable[T >: Null : TypeInfo](c: Code[T]): CodeNullable[T] =
    new CodeNullable(c)

  implicit def toCode[T](insn: => AbstractInsnNode): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      il += insn
    }
  }

  implicit def toCodeFromIndexedSeq[T](codes: => TraversableOnce[Code[T]]): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit =
      codes.foreach(_.emit(il))
  }

  implicit def indexedSeqValueToCode[T](v: IndexedSeq[Value[T]]): IndexedSeq[Code[T]] = v.map(_.get)

  implicit def valueToCode[T](v: Value[T]): Code[T] = v.get

  implicit def valueToCodeInt(f: Value[Int]): CodeInt = new CodeInt(f.get)

  implicit def valueToCodeLong(f: Value[Long]): CodeLong = new CodeLong(f.get)

  implicit def valueToCodeFloat(f: Value[Float]): CodeFloat = new CodeFloat(f.get)

  implicit def valueToCodeDouble(f: Value[Double]): CodeDouble = new CodeDouble(f.get)

  implicit def valueToCodeChar(f: Value[Char]): CodeChar = new CodeChar(f.get)

  implicit def valueToCodeString(f: Value[String]): CodeString = new CodeString(f.get)

  implicit def valueToCodeObject[T <: AnyRef](f: Value[T])(implicit tct: ClassTag[T]): CodeObject[T] = new CodeObject(f.get)

  implicit def valueToCodeArray[T](c: Value[Array[T]])(implicit tti: TypeInfo[T]): CodeArray[T] = new CodeArray(c)

  implicit def valueToCodeBoolean(f: Value[Boolean]): CodeBoolean = new CodeBoolean(f.get)

  implicit def valueToCodeNullable[T >: Null : TypeInfo](c: Value[T]): CodeNullable[T] = new CodeNullable(c)

  implicit def toCode[T](f: Settable[T]): Code[T] = f.load()

  implicit def toCodeInt(f: Settable[Int]): CodeInt = new CodeInt(f.load())

  implicit def toCodeLong(f: Settable[Long]): CodeLong = new CodeLong(f.load())

  implicit def toCodeFloat(f: Settable[Float]): CodeFloat = new CodeFloat(f.load())

  implicit def toCodeDouble(f: Settable[Double]): CodeDouble = new CodeDouble(f.load())

  implicit def toCodeChar(f: Settable[Char]): CodeChar = new CodeChar(f.load())

  implicit def toCodeString(f: Settable[String]): CodeString = new CodeString(f.load())

  implicit def toCodeArray[T](f: Settable[Array[T]])(implicit tti: TypeInfo[T]): CodeArray[T] = new CodeArray(f.load())

  implicit def toCodeBoolean(f: Settable[Boolean]): CodeBoolean = new CodeBoolean(f.load())

  implicit def toCodeObject[T <: AnyRef : ClassTag](f: Settable[T]): CodeObject[T] = new CodeObject[T](f.load())

  implicit def toCodeNullable[T >: Null : TypeInfo](f: Settable[T]): CodeNullable[T] = new CodeNullable[T](f.load())

  implicit def toLocalRefInt(f: LocalRef[Int]): LocalRefInt = new LocalRefInt(f)

  def _const[T](a: T): Value[T] = new Value[T] {
    def get: Code[T] = Code(new LdcInsnNode(a))
  }

  implicit def const(s: String): Value[String] = _const(s)

  implicit def const(b: Boolean): Value[Boolean] = _const(b)

  implicit def const(i: Int): Value[Int] = _const(i)

  implicit def const(l: Long): Value[Long] = _const(l)

  implicit def const(f: Float): Value[Float] = _const(f)

  implicit def const(d: Double): Value[Double] = _const(d)

  implicit def const(c: Char): Value[Char] = _const(c)

  implicit def const(b: Byte): Value[Byte] = _const(b)

  implicit def strToCode(s: String): Code[String] = _const(s)

  implicit def boolToCode(b: Boolean): Code[Boolean] = _const(b)

  implicit def intToCode(i: Int): Code[Int] = _const(i)

  implicit def longToCode(l: Long): Code[Long] = _const(l)

  implicit def floatToCode(f: Float): Code[Float] = _const(f)

  implicit def doubleToCode(d: Double): Code[Double] = _const(d)

  implicit def charToCode(c: Char): Code[Char] = _const(c)

  implicit def byteToCode(b: Byte): Code[Byte] = _const(b)
}
