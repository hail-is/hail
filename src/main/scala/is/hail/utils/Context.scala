package is.hail.utils

abstract class Context {
  def wrapException(e: Throwable): Nothing
}

case class TextContext(line: String, file: String, position: Option[Int]) extends Context {
  def wrapException(e: Throwable): Nothing = {
    val msg = e match {
      case _: FatalException => e.getMessage
      case _ => s"caught $e"
    }
    fatal(
      s"""$file${ position.map(ln => ":" + (ln + 1)).getOrElse("") }: $msg
         |  offending line: @1""".stripMargin, line)
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