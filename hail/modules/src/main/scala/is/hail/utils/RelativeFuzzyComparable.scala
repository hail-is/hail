package is.hail.utils

trait RelativeFuzzyComparable[A] {
  def relativeEq(tolerance: Double, x: A, y: A): Boolean
}

object RelativeFuzzyComparable {

  def relativeEq[T](tolerance: Double, x: T, y: T)(implicit afc: RelativeFuzzyComparable[T])
    : Boolean =
    afc.relativeEq(tolerance, x, y)

  implicit object rfcDoubles extends RelativeFuzzyComparable[Double] {
    def relativeEq(tolerance: Double, x: Double, y: Double) = D_==(x, y, tolerance)
  }

  implicit def rfcMaps[K, V](implicit vRFC: RelativeFuzzyComparable[V])
    : RelativeFuzzyComparable[Map[K, V]] =
    new RelativeFuzzyComparable[Map[K, V]] {
      def relativeEq(tolerance: Double, x: Map[K, V], y: Map[K, V]) =
        x.keySet == y.keySet && x.keys.forall(k => vRFC.relativeEq(tolerance, x(k), y(k)))
    }
}
