package is.hail.utils.richUtils

import is.hail.asm4s.{Code, LineNumber, TypeInfo}

class RichCodeIterator[T](it: Code[Iterator[T]]) {
  def hasNext(implicit line: LineNumber): Code[Boolean] = it.invoke[Boolean]("hasNext")
  def next()(implicit tti: TypeInfo[T], line: LineNumber): Code[T] =
    Code.checkcast[T](it.invoke[java.lang.Object]("next"))
}
