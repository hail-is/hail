package is.hail.utils

abstract class Context {
  def wrapException(e: Throwable): Nothing
}

case class TextContext(line: String, file: String, position: Option[Int]) extends Context {
  def wrapException(e: Throwable): Nothing = {
    e match {
      case _: HailException =>
        fatal(
          s"""$file${ position.map(ln => ":" + (ln + 1)).getOrElse("") }: ${ e.getMessage }
             |  offending line: @1""".stripMargin, line, e)
      case _ =>
        fatal(
          s"""$file${ position.map(ln => ":" + (ln + 1)).getOrElse("") }: caught ${ e.getClass.getName() }: ${ e.getMessage }
             |  offending line: @1""".stripMargin, line, e)
    }
  }
}

case class WithContext[T](value: T, source: Context) {
  def map[U](f: T => U): WithContext[U] = {
    try {
      copy[U](value = f(value))
    } catch {
      case e: Throwable => source.wrapException(e)
    }
  }

  def foreach(f: T => Unit) {
    try {
      f(value)
    } catch {
      case e: Exception => source.wrapException(e)
    }
  }
}