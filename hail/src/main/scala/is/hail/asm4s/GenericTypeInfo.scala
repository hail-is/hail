package is.hail.asm4s

import scala.language.implicitConversions
import scala.reflect.ClassTag

sealed abstract class MaybeGenericTypeInfo[T : TypeInfo] {
  def castFromGeneric(x: Code[_]): Code[T]
  def castToGeneric(x: Code[T]): Code[_]

  val base: TypeInfo[_]
  val generic: TypeInfo[_]
  val isGeneric: Boolean
}

final case class GenericTypeInfo[T : TypeInfo]() extends MaybeGenericTypeInfo[T] {
  val base = typeInfo[T]

  def castFromGeneric(_x: Code[_]) = {
    val x = _x.asInstanceOf[Code[AnyRef]]
    base match {
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
      case _: UnitInfo.type =>
        Code.toUnit(Code.checkcast[java.lang.Void](x)).asInstanceOf[Code[T]]
      case cti: ClassInfo[_] =>
        Code.checkcast[T](x)(cti.cct.asInstanceOf[ClassTag[T]])
      case ati: ArrayInfo[_] =>
        Code.checkcast[T](x)(ati.tct.asInstanceOf[ClassTag[T]])
    }
  }

  def castToGeneric(x: Code[T]) = base match {
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
    case _: UnitInfo.type =>
      Code(x, Code._null[java.lang.Void])
    case cti: ClassInfo[_] =>
      x
    case ati: ArrayInfo[_] =>
      x
  }

  val generic = classInfo[java.lang.Object]
  val isGeneric = true
}

final case class NotGenericTypeInfo[T : TypeInfo]() extends MaybeGenericTypeInfo[T] {
  def castFromGeneric(x: Code[_]): Code[T] = x.asInstanceOf[Code[T]]
  def castToGeneric(x: Code[T]): Code[_] = x

  val base = typeInfo[T]
  val generic = base
  val isGeneric = false
}
