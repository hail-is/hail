package is.hail.utils.richUtils

import scala.reflect.ClassTag

class RichAny(val a: Any) extends AnyVal {
  def castOption[T](implicit ct: ClassTag[T]): Option[T] =
    if (ct.runtimeClass.isInstance(a))
      Some(a.asInstanceOf[T])
    else
      None
}
