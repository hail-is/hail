package is.hail.asm4s

sealed abstract class MaybeGenericTypeInfo[T : TypeInfo] {
  def castFromGeneric(cb: CodeBuilderLike, x: Value[_]): Value[T]
  def castToGeneric(cb: CodeBuilderLike, x: Value[T]): Value[_]

  val base: TypeInfo[_]
  val generic: TypeInfo[_]
  val isGeneric: Boolean
}

final case class GenericTypeInfo[T : TypeInfo]() extends MaybeGenericTypeInfo[T] {
  val base = typeInfo[T]

  def castFromGeneric(cb: CodeBuilderLike, _x: Value[_]): Value[T] = {
    val x = coerce[AnyRef](_x)
    base match {
      case _: IntInfo.type =>
        coerce[T](cb.memoize(Code.intValue(Code.checkcast[java.lang.Integer](x))))
      case _: LongInfo.type =>
        coerce[T](cb.memoize(Code.longValue(Code.checkcast[java.lang.Long](x))))
      case _: FloatInfo.type =>
        coerce[T](cb.memoize(Code.floatValue(Code.checkcast[java.lang.Float](x))))
      case _: DoubleInfo.type =>
        coerce[T](cb.memoize(Code.doubleValue(Code.checkcast[java.lang.Double](x))))
      case _: ShortInfo.type =>
        coerce[T](cb.memoize(Code.checkcast[java.lang.Short](x).invoke[Short]("shortValue")))
      case _: ByteInfo.type =>
        coerce[T](cb.memoize(Code.checkcast[java.lang.Byte](x).invoke[Byte]("byteValue")))
      case _: BooleanInfo.type =>
        coerce[T](cb.memoize(Code.checkcast[java.lang.Boolean](x).invoke[Boolean]("booleanValue")))
      case _: CharInfo.type =>
        coerce[T](cb.memoize(Code.checkcast[java.lang.Character](x).invoke[Char]("charValue")))
      case _: UnitInfo.type =>
        coerce[T](Code._empty)
      case cti: ClassInfo[_] =>
        cb.memoize(Code.checkcast[T](x)(cti))
      case ati: ArrayInfo[_] =>
        cb.memoize(Code.checkcast[T](x)(ati))
    }
  }

  def castToGeneric(cb: CodeBuilderLike, x: Value[T]): Value[_] = base match {
    case _: IntInfo.type =>
      cb.memoize(Code.boxInt(coerce[Int](x)))
    case _: LongInfo.type =>
      cb.memoize(Code.boxLong(coerce[Long](x)))
    case _: FloatInfo.type =>
      cb.memoize(Code.boxFloat(coerce[Float](x)))
    case _: DoubleInfo.type =>
      cb.memoize(Code.boxDouble(coerce[Double](x)))
    case _: ShortInfo.type =>
      cb.memoize(Code.newInstance[java.lang.Short, Short](coerce[Short](x)))
    case _: ByteInfo.type =>
      cb.memoize(Code.newInstance[java.lang.Byte, Byte](coerce[Byte](x)))
    case _: BooleanInfo.type =>
      cb.memoize(Code.boxBoolean(coerce[Boolean](x)))
    case _: CharInfo.type =>
      cb.memoize(Code.newInstance[java.lang.Character, Char](coerce[Char](x)))
    case _: UnitInfo.type =>
      Code._null[java.lang.Void]
    case cti: ClassInfo[_] =>
      x
    case ati: ArrayInfo[_] =>
      x
  }

  val generic = classInfo[java.lang.Object]
  val isGeneric = true
}

final case class NotGenericTypeInfo[T : TypeInfo]() extends MaybeGenericTypeInfo[T] {
  def castFromGeneric(cb: CodeBuilderLike, x: Value[_]): Value[T] = coerce[T](x)
  def castToGeneric(cb: CodeBuilderLike, x: Value[T]): Value[_] = x

  val base = typeInfo[T]
  val generic = base
  val isGeneric = false
}
