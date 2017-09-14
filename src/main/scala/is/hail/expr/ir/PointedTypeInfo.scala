package is.hail.expr.ir

import is.hail.asm4s._

sealed trait PointedTypeInfo[T] {
  def ti: TypeInfo[T]
  def point: Code[_]
}
object PointedTypeInfo {
  def pointedTypeInfo[T](implicit tpti: PointedTypeInfo[T]) = tpti
  implicit case object PointedIntTypeInfo extends PointedTypeInfo[Int] {
    val ti = IntInfo
    val point = const(-2)
  }
  implicit case object PointedLongTypeInfo extends PointedTypeInfo[Long] {
    val ti = LongInfo
    val point = const(-3L)
  }
  implicit case object PointedFloatTypeInfo extends PointedTypeInfo[Float] {
    val ti = FloatInfo
    val point = const(-4.0f)
  }
  implicit case object PointedDoubleTypeInfo extends PointedTypeInfo[Double] {
    val ti = DoubleInfo
    val point = const(-5.0)
  }
  implicit case object PointedAnyRefTypeInfo extends PointedTypeInfo[AnyRef] {
    val ti = classInfo[AnyRef]
    val point = Code._null
  }
}

