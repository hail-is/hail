package is.hail.utils.implicits

class RichPartialKleisliOptionFunction[A, B](pf: PartialFunction[A, Option[B]]) {
  def flatLift: A => Option[B] = a => pf.lift(a).flatten
}
