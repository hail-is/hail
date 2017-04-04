package is.hail.utils

import scala.math.Numeric.{DoubleIsFractional, FloatIsFractional, IntIsIntegral, LongIsIntegral}

trait NumericPair[T, U] extends Serializable {

  def box(t: T): U

  def unbox(u: U): T

  def numeric: scala.math.Numeric[T]
}

object NumericPairs {

  implicit object IntPair extends NumericPair[Int, java.lang.Integer] {

    def box(t: Int): java.lang.Integer = Int.box(t)

    def unbox(u: java.lang.Integer): Int = Int.unbox(u)

    def numeric: scala.math.Numeric[Int] = IntIsIntegral
  }

  implicit object LongPair extends NumericPair[Long, java.lang.Long] {

    def box(t: Long): java.lang.Long = Long.box(t)

    def unbox(u: java.lang.Long): Long = Long.unbox(u)

    def numeric: scala.math.Numeric[Long] = LongIsIntegral
  }

  implicit object FloatPair extends NumericPair[Float, java.lang.Float] {

    def box(t: Float): java.lang.Float = Float.box(t)

    def unbox(u: java.lang.Float): Float = Float.unbox(u)

    def numeric: scala.math.Numeric[Float] = FloatIsFractional
  }


  implicit object DoublePair extends NumericPair[Double, java.lang.Double] {

    def box(t: Double): java.lang.Double = Double.box(t)

    def unbox(u: java.lang.Double): Double = Double.unbox(u)

    def numeric: scala.math.Numeric[Double] = DoubleIsFractional
  }
}
