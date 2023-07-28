package is.hail.utils.richUtils

case class RichVal[A](a: A) extends AnyVal {
  def &[B](f: A => B): B =
    f(a)
}
