package is.hail.utils.richUtils

class RichPartialKleisliOptionFunction[A, B](pf: PartialFunction[A, Option[B]]) {
  def flatLift: A => Option[B] = a => pf.lift(a).flatten
}
