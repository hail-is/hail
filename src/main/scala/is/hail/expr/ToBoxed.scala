package is.hail.expr

sealed trait ToBoxed[T] {
  type T <: AnyRef
}
sealed trait BoxedBoolean extends ToBoxed[Boolean] {
  type T = java.lang.Boolean
}
sealed trait BoxedByte extends ToBoxed[Byte] {
  type T = java.lang.Byte
}
sealed trait BoxedShort extends ToBoxed[Short] {
  type T = java.lang.Short
}
sealed trait BoxedInt extends ToBoxed[Int] {
  type T = java.lang.Integer
}
sealed trait BoxedLong extends ToBoxed[Long] {
  type T = java.lang.Long
}
sealed trait BoxedFloat extends ToBoxed[Float] {
  type T = java.lang.Float
}
sealed trait BoxedDouble extends ToBoxed[Double] {
  type T = java.lang.Double
}
