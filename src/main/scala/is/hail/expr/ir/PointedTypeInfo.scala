package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.asm4s.ucode.UCode

sealed trait PointedTypeInfo[T] {
  def ti: TypeInfo[T]
  def point: UCode[_]
}
object PointedTypeInfo {
  def pointedTypeInfo[T](implicit tpti: PointedTypeInfo[T]) = tpti
  implicit case object PointedIntTypeInfo extends PointedTypeInfo[Int] {
    val ti = IntInfo
    val point = ucode.I32(-2)
  }
  implicit case object PointedLongTypeInfo extends PointedTypeInfo[Long] {
    val ti = LongInfo
    val point = ucode.I64(-3)
  }
  implicit case object PointedFloatTypeInfo extends PointedTypeInfo[Float] {
    val ti = FloatInfo
    val point = ucode.F32(-4.0f)
  }
  implicit case object PointedDoubleTypeInfo extends PointedTypeInfo[Double] {
    val ti = DoubleInfo
    val point = ucode.F64(-5.0)
  }
  implicit case object PointedAnyRefTypeInfo extends PointedTypeInfo[AnyRef] {
    val ti = classInfo[AnyRef]
    val point = ucode.Null()
  }
}

