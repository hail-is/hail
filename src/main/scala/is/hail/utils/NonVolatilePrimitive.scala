package is.hail.utils

sealed trait NonVolatilePrimitive[T] { }

// essentially, these must be non-mutable values, ergo, if they're in a Volatile
// collection, the collection may be treated as non volatile
object NonVolatilePrimitive {
  implicit object StringNonVolatilePrimitive extends NonVolatilePrimitive[String]
  implicit object IntNonVolatilePrimitive extends NonVolatilePrimitive[Int]
  implicit object LongNonVolatilePrimitive extends NonVolatilePrimitive[Long]
  implicit object FloatNonVolatilePrimitive extends NonVolatilePrimitive[Float]
  implicit object DoubleNonVolatilePrimitive extends NonVolatilePrimitive[Double]
}
