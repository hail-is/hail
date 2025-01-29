package is.hail.utils

trait NumericPair[T, BoxedT >: Null] extends Serializable {

  def box(t: T): BoxedT

  def unbox(u: BoxedT): T

  implicit val numeric: scala.math.Numeric[T]
}

trait NumericPairImplicits {

  implicit object IntPair extends NumericPair[Int, java.lang.Integer] {

    def box(t: Int): java.lang.Integer = Int.box(t)

    def unbox(u: java.lang.Integer): Int = Int.unbox(u)

    val numeric: scala.math.Numeric[Int] = implicitly[Numeric[Int]]
  }

  implicit object LongPair extends NumericPair[Long, java.lang.Long] {

    def box(t: Long): java.lang.Long = Long.box(t)

    def unbox(u: java.lang.Long): Long = Long.unbox(u)

    val numeric: scala.math.Numeric[Long] = implicitly[Numeric[Long]]
  }

  implicit object FloatPair extends NumericPair[Float, java.lang.Float] {

    def box(t: Float): java.lang.Float = Float.box(t)

    def unbox(u: java.lang.Float): Float = Float.unbox(u)

    val numeric: scala.math.Numeric[Float] = implicitly[Numeric[Float]]
  }

  implicit object DoublePair extends NumericPair[Double, java.lang.Double] {

    def box(t: Double): java.lang.Double = Double.box(t)

    def unbox(u: java.lang.Double): Double = Double.unbox(u)

    val numeric: scala.math.Numeric[Double] = implicitly[Numeric[Double]]
  }
}
