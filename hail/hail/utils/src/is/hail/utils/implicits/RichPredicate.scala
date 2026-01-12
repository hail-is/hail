package is.hail.utils.implicits

class RichPredicate[A](val f: A => Boolean) extends AnyVal {
  def and[B >: A](g: B => Boolean): A => Boolean =
    a => f(a) && g(a)

  def or[B >: A](g: B => Boolean): A => Boolean =
    a => f(a) || g(a)

  def xor[B >: A](g: B => Boolean): A => Boolean =
    a => f(a) ^ g(a)
}
