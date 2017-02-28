package is.hail.utils

class HailException(val msg: String, val logMsg: Option[String] = None) extends RuntimeException(msg)

class UserException(val msg: String, val logMsg: Option[String] = None) extends RuntimeException(msg)

trait ErrorHandling {

  def abort(msg: String): Nothing = throw new UserException(msg)

  def abort(msg: String, t: Truncatable): Nothing = {
    val (screen, logged) = t.strings
    throw new UserException(format(msg, screen), Some(format(msg, logged)))
  }

  def abort(msg: String, t1: Truncatable, t2: Truncatable): Nothing = {
    val (screen1, logged1) = t1.strings
    val (screen2, logged2) = t2.strings
    throw new UserException(format(msg, screen1, screen2), Some(format(msg, logged1, logged2)))
  }

  def fatal(msg: String) : Nothing = throw new HailException(msg)

  def fatal(msg: String, t: Truncatable): Nothing = {
    val (screen, logged) = t.strings
    throw new HailException(format(msg, screen), Some(format(msg, logged)))
  }

  def fatal(msg: String, t1: Truncatable, t2: Truncatable): Nothing = {
    val (screen1, logged1) = t1.strings
    val (screen2, logged2) = t2.strings
    throw new HailException(format(msg, screen1, screen2), Some(format(msg, logged1, logged2)))
  }

  def wrappedUserException(e: Throwable): Option[UserException] = {
    e match {
      case ue: UserException =>
        Some(ue)
      case _ =>
        Option(e.getCause).flatMap(c => wrappedUserException(c))
    }
  }

  def deepestMessage(e: Throwable): String = {
    var iterE = e
    while (iterE.getCause != null)
      iterE = iterE.getCause

    s"${ iterE.getClass.getSimpleName }: ${ iterE.getLocalizedMessage }"
  }

  def expandException(e: Throwable, logMessage: Boolean): String = {
    val msg = e match {
      case e: HailException => e.logMsg.filter(_ => logMessage).getOrElse(e.msg)
      case e: UserException => e.logMsg.filter(_ => logMessage).getOrElse(e.msg)
      case _ => e.getLocalizedMessage
    }
    s"\n${ e.getClass.getName }: $msg\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).map(exception => expandException(exception, logMessage)).getOrElse("")
    }"
  }

  def getMinimalMessage(e: Throwable): String = {
    wrappedUserException(e) match {
      case Some(userException) =>
        log.error(s"hail: fatal: ${ userException.logMsg }\nFrom ${ expandException(e, logMessage = true) }")
        userException.msg
      case None =>
        val msg = deepestMessage(e)
        log.error(s"hail: caught exception: $msg\nFrom ${
          expandException(e,
            logMessage = true)
        }")
        s"$msg\nFrom ${ expandException(e, logMessage = false) }"
    }
  }
}
