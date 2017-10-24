package is.hail

import java.io.PrintWriter
import java.lang.reflect.Method

import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect

package object asm4s {

  def typeInfo[T](implicit tti: TypeInfo[T]): TypeInfo[T] = tti

  abstract class TypeInfo[T : ClassTag] {
    val name: String
    val iname: String = name
    val loadOp: Int
    val storeOp: Int
    val aloadOp: Int
    val astoreOp: Int
    val returnOp: Int
    val slots: Int = 1

    def newArray(): AbstractInsnNode
    def classTag: ClassTag[T] = reflect.classTag
  }

  sealed abstract class MaybeGenericTypeInfo[T : TypeInfo] extends TypeInfo[T]()(typeInfo[T].classTag) {
    def castFromGeneric: Code[_] => Code[T]
    def castToGeneric: Code[T] => Code[_]

    val generic: TypeInfo[_]
    val isGeneric: Boolean
  }

  final case class GenericTypeInfo[T : TypeInfo]() extends MaybeGenericTypeInfo[T] {
    private val ti = typeInfo[T]

    val name = ti.name
    val loadOp = ti.loadOp
    val storeOp = ti.storeOp
    val aloadOp = ti.aloadOp
    val astoreOp = ti.astoreOp
    val returnOp = ti.returnOp
    override val slots = ti.slots

    def newArray() = ti.newArray

    def castFromGeneric = (_x: Code[_]) => {
      val x = _x.asInstanceOf[Code[AnyRef]]
      ti match {
        case _: IntInfo.type =>
          Code.intValue(Code.checkcast[java.lang.Integer](x)).asInstanceOf[Code[T]]
        case _: LongInfo.type =>
          Code.longValue(Code.checkcast[java.lang.Long](x)).asInstanceOf[Code[T]]
        case _: FloatInfo.type =>
          Code.floatValue(Code.checkcast[java.lang.Float](x)).asInstanceOf[Code[T]]
        case _: DoubleInfo.type =>
          Code.doubleValue(Code.checkcast[java.lang.Double](x)).asInstanceOf[Code[T]]
        case _: ShortInfo.type =>
          Code.checkcast[java.lang.Short](x).invoke[Short]("shortValue").asInstanceOf[Code[T]]
        case _: ByteInfo.type =>
          Code.checkcast[java.lang.Byte](x).invoke[Byte]("byteValue").asInstanceOf[Code[T]]
        case _: BooleanInfo.type =>
          Code.checkcast[java.lang.Boolean](x).invoke[Boolean]("booleanValue").asInstanceOf[Code[T]]
        case _: CharInfo.type =>
          Code.checkcast[java.lang.Character](x).invoke[Char]("charValue").asInstanceOf[Code[T]]
        case cti: ClassInfo[_] =>
          Code.checkcast[T](x)(cti.cct.asInstanceOf[ClassTag[T]])
        case ati: ArrayInfo[_] =>
          Code.checkcast[T](x)(ati.tct.asInstanceOf[ClassTag[T]])
      }
    }

    def castToGeneric = (x: Code[T]) => ti match {
      case _: IntInfo.type =>
        Code.boxInt(x.asInstanceOf[Code[Int]])
      case _: LongInfo.type =>
        Code.boxLong(x.asInstanceOf[Code[Long]])
      case _: FloatInfo.type =>
        Code.boxFloat(x.asInstanceOf[Code[Float]])
      case _: DoubleInfo.type =>
        Code.boxDouble(x.asInstanceOf[Code[Double]])
      case _: ShortInfo.type =>
        Code.newInstance[java.lang.Short, Short](x.asInstanceOf[Code[Short]])
      case _: ByteInfo.type =>
        Code.newInstance[java.lang.Byte, Byte](x.asInstanceOf[Code[Byte]])
      case _: BooleanInfo.type =>
        Code.boxBoolean(x.asInstanceOf[Code[Boolean]])
      case _: CharInfo.type =>
        Code.newInstance[java.lang.Character, Char](x.asInstanceOf[Code[Char]])
      case cti: ClassInfo[_] =>
        x
      case ati: ArrayInfo[_] =>
        x
    }

    val generic = classInfo[java.lang.Object]
    val isGeneric = true

  }

  final case class NotGenericTypeInfo[T : TypeInfo]() extends MaybeGenericTypeInfo[T] {
    private val ti = typeInfo[T]

    val name = ti.name
    val loadOp = ti.loadOp
    val storeOp = ti.storeOp
    val aloadOp = ti.aloadOp
    val astoreOp = ti.astoreOp
    val returnOp = ti.returnOp
    override val slots = ti.slots

    def newArray() = ti.newArray

    def castFromGeneric = (x: Code[_]) => x.asInstanceOf[Code[T]]
    def castToGeneric = (x: Code[_]) => x

    val generic = ti
    val isGeneric = false

  }

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

  implicit def classInfo[C <: AnyRef](implicit cct: ClassTag[C]): TypeInfo[C] =
    new ClassInfo

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

  implicit def arrayInfo[T](implicit cct: ClassTag[Array[T]]): TypeInfo[Array[T]] =
    new ArrayInfo

  implicit def arrayInfo[T](implicit tti: TypeInfo[T]): TypeInfo[Array[T]] =
    new ArrayInfo(tti)

  class ArrayInfo[T](implicit val tct: ClassTag[Array[T]]) extends TypeInfo[Array[T]] {
    def this(tti: TypeInfo[T]) = this()(tti.classTag.wrap)

    val name = Type.getDescriptor(tct.runtimeClass)
    override val iname = Type.getInternalName(tct.runtimeClass)
    val loadOp = ALOAD
    val storeOp = ASTORE
    val aloadOp = AALOAD
    val astoreOp = AASTORE
    val returnOp = ARETURN

    def newArray() = new TypeInsnNode(ANEWARRAY, iname)
  }

  object HailClassLoader extends ClassLoader {
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

  def loadClass(className: String, b: Array[Byte]): Class[_] = {
    HailClassLoader.loadOrDefineClass(className, b)
  }

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

  implicit def toCodeArray[T](c: Code[Array[T]])(implicit tti: TypeInfo[T]): CodeArray[T] = new CodeArray(c)

  implicit def toCodeObject[T >: Null](c: Code[T])(implicit tti: TypeInfo[T], tct: ClassTag[T]): CodeObject[T] =
    new CodeObject(c)

  implicit def toCode[T](insn: => AbstractInsnNode): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit = {
      il += insn
    }
  }

  implicit def toCodeFromIndexedSeq[T](codes: => TraversableOnce[Code[T]]): Code[T] = new Code[T] {
    def emit(il: Growable[AbstractInsnNode]): Unit =
      codes.foreach(_.emit(il))
  }

  implicit def toCode[T](f: LocalRef[T]): Code[T] = f.load()

  implicit def toCodeInt(f: LocalRef[Int]): CodeInt = new CodeInt(f.load())

  implicit def toCodeLong(f: LocalRef[Long]): CodeLong = new CodeLong(f.load())

  implicit def toCodeFloat(f: LocalRef[Float]): CodeFloat = new CodeFloat(f.load())

  implicit def toCodeDouble(f: LocalRef[Double]): CodeDouble = new CodeDouble(f.load())

  implicit def toCodeArray[T](f: LocalRef[Array[T]])(implicit tti: TypeInfo[T]): CodeArray[T] = new CodeArray(f.load())

  implicit def toCodeBoolean(f: LocalRef[Boolean]): CodeBoolean = new CodeBoolean(f.load())

  implicit def toCodeObject[T >: Null](f: LocalRef[T])(implicit tti: TypeInfo[T], tct: ClassTag[T]): CodeObject[T] = new CodeObject[T](f.load())

  implicit def toLocalRefInt(f: LocalRef[Int]): LocalRefInt = new LocalRefInt(f)

  implicit def const(s: String): Code[String] = Code(new LdcInsnNode(s))

  implicit def const(b: Boolean): Code[Boolean] = Code(new LdcInsnNode(if (b) 1 else 0))

  implicit def const(i: Int): Code[Int] = Code(new LdcInsnNode(i))

  implicit def const(l: Long): Code[Long] = Code(new LdcInsnNode(l))

  implicit def const(f: Float): Code[Float] = Code(new LdcInsnNode(f))

  implicit def const(d: Double): Code[Double] = Code(new LdcInsnNode(d))

  implicit def const(b: Byte): Code[Byte] = Code(new LdcInsnNode(b))
}
