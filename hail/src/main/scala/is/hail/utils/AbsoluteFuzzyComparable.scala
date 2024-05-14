package is.hail.utils

trait AbsoluteFuzzyComparable[A] {
  def absoluteEq(tolerance: Double, x: A, y: A): Boolean
}

object AbsoluteFuzzyComparable {

  def absoluteEq[T](tolerance: Double, x: T, y: T)(implicit afc: AbsoluteFuzzyComparable[T])
    : Boolean =
    afc.absoluteEq(tolerance, x, y)

  implicit object afcDoubles extends AbsoluteFuzzyComparable[Double] {
    def absoluteEq(tolerance: Double, x: Double, y: Double) = Math.abs(x - y) <= tolerance
  }

  implicit def afcMaps[K, V](implicit vRFC: AbsoluteFuzzyComparable[V])
    : AbsoluteFuzzyComparable[Map[K, V]] =
    new AbsoluteFuzzyComparable[Map[K, V]] {
      def absoluteEq(tolerance: Double, x: Map[K, V], y: Map[K, V]) =
        x.keySet == y.keySet && x.keys.forall(k => vRFC.absoluteEq(tolerance, x(k), y(k)))
    }
}
