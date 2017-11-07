package is.hail.utils

class HailException(val msg: String, val logMsg: Option[String] = None, cause: Throwable = null) extends RuntimeException(msg, cause)

trait ErrorHandling {
  def fatal(msg: String): Nothing = throw new HailException(msg)

  def fatal(msg: String, t: Truncatable): Nothing =
    fatal(msg, t, null: Throwable)

  def fatal(msg: String, t: Truncatable, e: Throwable): Nothing = {
    val (screen, logged) = t.strings
    throw new HailException(format(msg, screen), Some(format(msg, logged)), e)
  }

  def fatal(msg: String, t1: Truncatable, t2: Truncatable): Nothing =
    fatal(msg, t1, t2, null)

  def fatal(msg: String, t1: Truncatable, t2: Truncatable, e: Throwable): Nothing = {
    val (screen1, logged1) = t1.strings
    val (screen2, logged2) = t2.strings
    throw new HailException(format(msg, screen1, screen2), Some(format(msg, logged1, logged2)), e)
  }

  def deepestMessage(e: Throwable): String = {
    var iterE = e
    while (iterE.getCause != null)
      iterE = iterE.getCause

    s"${ iterE.getClass.getSimpleName }: ${ iterE.getMessage }"
  }

  def expandException(e: Throwable, logMessage: Boolean): String = {
    val msg = e match {
      case e: HailException => e.logMsg.filter(_ => logMessage).getOrElse(e.msg)
      case _ => e.getLocalizedMessage
    }
    s"${ e.getClass.getName }: $msg\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).map(exception => expandException(exception, logMessage)).getOrElse("")
    }\n"
  }

  def handleForPython(e: Throwable): (String, String) = {
    val short = deepestMessage(e)
    val expanded = expandException(e, false)
    val logExpanded = expandException(e, true)

    log.error(s"$short\nFrom $logExpanded")

    (short, expanded)
  }
}
