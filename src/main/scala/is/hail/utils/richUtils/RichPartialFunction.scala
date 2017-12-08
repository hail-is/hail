package is.hail.utils.richUtils

class RichPartialFunction[A, B](pf: PartialFunction[A, B]) {
  def flatten[C](implicit isOption: B =:= Option[C]): PartialFunction[A, C] = new PartialFunction[A, C] {
    def isDefinedAt(a: A): Boolean = pf.isDefinedAt(a) && pf(a).isDefined
    def apply(a: A): C = pf(a).get
  }
}
