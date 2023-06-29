package is.hail

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.language.implicitConversions
import scala.reflect.ClassTag

package asm4s {
  trait TypeInfo[T] {
    val desc: String
    val iname: String = desc
    def loadOp: Int
    def storeOp: Int
    def aloadOp: Int
    def astoreOp: Int
    val returnOp: Int
    def slots: Int = 1

    def newArray(): AbstractInsnNode

    override def hashCode(): Int = desc.hashCode()

    override def equals(that: Any): Boolean = that match {
      case that: TypeInfo[_] =>
        desc == that.desc
      case _ => false
    }

    override def toString: String = desc

    def uninitializedValue: Value[_]
  }

  class ClassInfo[C](className: String) extends TypeInfo[C] {
    val desc = s"L${ className.replace(".", "/") };"
    override val iname = className.replace(".", "/")
    val loadOp = ALOAD
    val storeOp = ASTORE
    val aloadOp = AALOAD
    val astoreOp = AASTORE
    val returnOp = ARETURN

    def newArray(): AbstractInsnNode = new TypeInsnNode(ANEWARRAY, iname)

    override def uninitializedValue: Value[_] = Code._uncheckednull(this)
  }

  class ArrayInfo[T](implicit val tti: TypeInfo[T]) extends TypeInfo[Array[T]] {
    val desc = s"[${ tti.desc }"
    override val iname = desc.replace(".", "/")
    val loadOp = ALOAD
    val storeOp = ASTORE
    val aloadOp = AALOAD
    val astoreOp = AASTORE
    val returnOp = ARETURN

    def newArray() = new TypeInsnNode(ANEWARRAY, iname)

    override def uninitializedValue: Value[_] = Code._null[Array[T]](this)
  }
}

package object asm4s {
  lazy val theHailClassLoaderForSparkWorkers = {
    // FIXME: how do I ensure this is only created in Spark workers?
    new HailClassLoader(getClass().getClassLoader())
  }

  def genName(tag: String, baseName: String): String = lir.genName(tag, baseName)

  def typeInfo[T](implicit tti: TypeInfo[T]): TypeInfo[T] = tti

  def coerce[T](c: Code[_]): Code[T] =
    c.asInstanceOf[Code[T]]

  def coerce[T](c: Value[_]): Value[T] =
    c.asInstanceOf[Value[T]]

  def coerce[T](c: Settable[_]): Settable[T] =
    c.asInstanceOf[Settable[T]]

  def typeInfoFromClassTag[T](ct: ClassTag[T]): TypeInfo[T] =
    typeInfoFromClass(ct.runtimeClass.asInstanceOf[Class[T]])

  def typeInfoFromClass[T](c: Class[T]): TypeInfo[T] = {
    val ti: TypeInfo[_] = if (c.isPrimitive) {
      if (c == java.lang.Void.TYPE)
        UnitInfo
      else if (c == java.lang.Byte.TYPE)
        ByteInfo
      else if (c == java.lang.Short.TYPE)
        ShortInfo
      else if (c == java.lang.Boolean.TYPE)
        BooleanInfo
      else if (c == java.lang.Integer.TYPE)
        IntInfo
      else if (c == java.lang.Long.TYPE)
        LongInfo
      else if (c == java.lang.Float.TYPE)
        FloatInfo
      else if (c == java.lang.Double.TYPE)
        DoubleInfo
      else {
        assert(c == java.lang.Character.TYPE, c)
        CharInfo
      }
    } else if (c.isArray) {
      arrayInfo(typeInfoFromClass(c.getComponentType))
    } else
      classInfoFromClass(c)
    ti.asInstanceOf[TypeInfo[T]]
  }

  implicit object BooleanInfo extends TypeInfo[Boolean] {
    val desc = "Z"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = BALOAD
    val astoreOp = BASTORE
    val returnOp = IRETURN
    val newarrayOp = NEWARRAY

    def newArray() = new IntInsnNode(NEWARRAY, T_BOOLEAN)

    override def uninitializedValue: Value[_] = const(false)
  }

  implicit object ByteInfo extends TypeInfo[Byte] {
    val desc = "B"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = BALOAD
    val astoreOp = BASTORE
    val returnOp = IRETURN
    val newarrayOp = NEWARRAY

    def newArray() = new IntInsnNode(NEWARRAY, T_BYTE)

    override def uninitializedValue: Value[_] = const(0.toByte)
  }

  implicit object ShortInfo extends TypeInfo[Short] {
    val desc = "S"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN
    val newarrayOp = NEWARRAY

    def newArray() = new IntInsnNode(NEWARRAY, T_SHORT)

    override def uninitializedValue: Value[_] = const(0.toShort)
  }

  implicit object IntInfo extends TypeInfo[Int] {
    val desc = "I"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN

    def newArray() = new IntInsnNode(NEWARRAY, T_INT)

    override def uninitializedValue: Value[_] = const(0)
  }

  implicit object LongInfo extends TypeInfo[Long] {
    val desc = "J"
    val loadOp = LLOAD
    val storeOp = LSTORE
    val aloadOp = LALOAD
    val astoreOp = LASTORE
    val returnOp = LRETURN
    override val slots = 2

    def newArray() = new IntInsnNode(NEWARRAY, T_LONG)

    override def uninitializedValue: Value[_] = const(0L)
  }

  implicit object FloatInfo extends TypeInfo[Float] {
    val desc = "F"
    val loadOp = FLOAD
    val storeOp = FSTORE
    val aloadOp = FALOAD
    val astoreOp = FASTORE
    val returnOp = FRETURN


    def newArray() = new IntInsnNode(NEWARRAY, T_FLOAT)

    override def uninitializedValue: Value[_] = const(0f)
  }

  implicit object DoubleInfo extends TypeInfo[Double] {
    val desc = "D"
    val loadOp = DLOAD
    val storeOp = DSTORE
    val aloadOp = DALOAD
    val astoreOp = DASTORE
    val returnOp = DRETURN
    override val slots = 2

    def newArray() = new IntInsnNode(NEWARRAY, T_DOUBLE)

    override def uninitializedValue: Value[_] = const(0d)
  }

  implicit object CharInfo extends TypeInfo[Char] {
    val desc = "C"
    val loadOp = ILOAD
    val storeOp = ISTORE
    val aloadOp = IALOAD
    val astoreOp = IASTORE
    val returnOp = IRETURN
    override val slots = 2

    def newArray() = new IntInsnNode(NEWARRAY, T_CHAR)

    override def uninitializedValue: Value[_] = const(0.toChar)
  }

  implicit object UnitInfo extends TypeInfo[Unit] {
    val desc = "V"
    def loadOp = ???
    def storeOp = ???
    def aloadOp = ???
    def astoreOp = ???
    val returnOp = RETURN
    override def slots = ???

    def newArray() = ???

    override def uninitializedValue: Value[_] = Code._empty
  }

  def classInfoFromClass[C](c: Class[C]): ClassInfo[C] = {
    assert(!c.isPrimitive && !c.isArray)
    new ClassInfo[C](c.getName)
  }

  implicit def classInfo[C <: AnyRef](implicit cct: ClassTag[C]): TypeInfo[C] =
    new ClassInfo[C](cct.runtimeClass.getName)

  implicit def arrayInfo[T](implicit tti: TypeInfo[T]): TypeInfo[Array[T]] =
    new ArrayInfo

  def loadClass(hcl: HailClassLoader, className: String, b: Array[Byte]): Class[_] =
    hcl.loadOrDefineClass(className, b)

  def loadClass(hcl: HailClassLoader, className: String): Class[_] =
    hcl.loadClass(className)

  def ??? = throw new UnsupportedOperationException

  implicit def toCodeBoolean(c: Code[Boolean]): CodeBoolean = new CodeBoolean(c)

  implicit def toCodeInt(c: Code[Int]): CodeInt = new CodeInt(c)

  implicit def byteToCodeInt(c: Code[Byte]): Code[Int] = c.asInstanceOf[Code[Int]]

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

  def _const[T](a: Any, ti: TypeInfo[T]): Value[T] =
    Value.fromLIR[T](lir.ldcInsn(a, ti))

  implicit def const(s: String): Value[String] = {
    require(s != null)
    _const(s, classInfo[String])
  }

  implicit def const(b: Boolean): Value[Boolean] = new Value[Boolean] {
    def get: Code[Boolean] = new ConstCodeBoolean(b)
  }

  implicit def const(i: Int): Value[Int] = _const(i, IntInfo)

  implicit def const(l: Long): Value[Long] = _const(l, LongInfo)

  implicit def const(f: Float): Value[Float] = _const(f, FloatInfo)

  implicit def const(d: Double): Value[Double] = _const(d, DoubleInfo)

  implicit def const(c: Char): Value[Char] = _const(c.toInt, CharInfo)

  implicit def const(b: Byte): Value[Byte] = _const(b.toInt, ByteInfo)

  implicit def strToCode(s: String): Code[String] = const(s)

  implicit def boolToCode(b: Boolean): Code[Boolean] = const(b)

  implicit def intToCode(i: Int): Code[Int] = const(i)

  implicit def longToCode(l: Long): Code[Long] = const(l)

  implicit def floatToCode(f: Float): Code[Float] = const(f)

  implicit def doubleToCode(d: Double): Code[Double] = const(d)

  implicit def charToCode(c: Char): Code[Char] = const(c)

  implicit def byteToCode(b: Byte): Code[Byte] = const(b)
}
