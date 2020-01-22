package is.hail.utils.richUtils

import is.hail.asm4s.Code
import scala.reflect.ClassTag

class RichCodeIterator[T](it: Code[Iterator[T]]) {
  def hasNext: Code[Boolean] = it.invoke[Boolean]("hasNext")
  def next()(implicit ct: ClassTag[T]): Code[T] =
    Code.checkcast[T](it.invoke[java.lang.Object]("next"))
}
