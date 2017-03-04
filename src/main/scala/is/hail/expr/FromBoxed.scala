package is.hail.expr

sealed trait FromBoxed[T] {
  type T
}

sealed trait UnBoxedBoolean extends FromBoxed[java.lang.Boolean] {
  type T = Boolean
}
sealed trait UnBoxedByte extends FromBoxed[java.lang.Byte] {
  type T = Byte
}
sealed trait UnBoxedShort extends FromBoxed[java.lang.Short] {
  type T = Short
}
sealed trait UnBoxedInt extends FromBoxed[java.lang.Integer] {
  type T = Int
}
sealed trait UnBoxedLong extends FromBoxed[java.lang.Long] {
  type T = Long
}
sealed trait UnBoxedFloat extends FromBoxed[java.lang.Float] {
  type T = Float
}
sealed trait UnBoxedDouble extends FromBoxed[java.lang.Double] {
  type T = Double
}
