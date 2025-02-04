package is.hail.utils.richUtils

class RichEnumeration[T <: Enumeration](val e: T) extends AnyVal {
  def withNameOption(name: String): Option[T#Value] =
    e.values.find(_.toString == name)
}
