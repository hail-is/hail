package is.hail.utils.richUtils

import is.hail.asm4s.{Code, TypeInfo}

class RichCodeIterator[T](it: Code[Iterator[T]]) {
  def hasNext: Code[Boolean] = it.invoke[Boolean]("hasNext")

  def next()(implicit tti: TypeInfo[T]): Code[T] =
    Code.checkcast[T](it.invoke[java.lang.Object]("next"))
}
