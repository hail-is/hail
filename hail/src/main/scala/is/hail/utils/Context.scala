package is.hail.utils

case class Context(line: String, file: String, position: Option[Int]) {
  def locationString: String =
    position match {
      case Some(p) => s"$file:$p"
      case None => file
    }

  def locationString(col: Int): String =
    position match {
      case Some(p) => s"$file:$p.$col"
      case None => s"$file:column $col"
    }

  def wrapException(e: Throwable): Nothing = {
    e match {
      case _: HailException =>
        fatal(
          s"""$locationString: ${e.getMessage}
             |  offending line: @1""".stripMargin,
          line,
          e,
        )
      case _ =>
        fatal(
          s"""$locationString: caught ${e.getClass.getName}: ${e.getMessage}
             |  offending line: @1""".stripMargin,
          line,
          e,
        )
    }
  }
}

case class WithContext[T](value: T, source: Context) {
  def map[U](f: T => U): WithContext[U] =
    try
      copy[U](value = f(value))
    catch {
      case e: Throwable => source.wrapException(e)
    }

  def wrap[U](f: T => U): U =
    try
      f(value)
    catch {
      case e: Throwable => source.wrapException(e)
    }

  def foreach(f: T => Unit): Unit = {
    try
      f(value)
    catch {
      case e: Exception => source.wrapException(e)
    }
  }
}
