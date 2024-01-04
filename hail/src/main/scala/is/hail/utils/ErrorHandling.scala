package is.hail.utils

import scala.annotation.tailrec

class HailException(val msg: String, val logMsg: Option[String], cause: Throwable, val errorId: Int)
    extends RuntimeException(msg, cause) {
  def this(msg: String) = this(msg, None, null, -1)
  def this(msg: String, logMsg: Option[String]) = this(msg, logMsg, null, -1)
  def this(msg: String, logMsg: Option[String], cause: Throwable) = this(msg, logMsg, cause, -1)
  def this(msg: String, errorId: Int) = this(msg, None, null, errorId)
}

class HailWorkerException(
  val partitionId: Int,
  val shortMessage: String,
  val expandedMessage: String,
  val errorId: Int,
) extends RuntimeException(s"[partitionId=$partitionId] " + shortMessage)

trait ErrorHandling {
  def fatal(msg: String): Nothing = throw new HailException(msg)

  def fatal(msg: String, errorId: Int) = throw new HailException(msg, errorId)

  def fatal(msg: String, cause: Throwable): Nothing = throw new HailException(msg, None, cause)

  def fatal(msg: String, t: Truncatable): Nothing = {
    val (screen, logged) = t.strings
    throw new HailException(format(msg, screen), Some(format(msg, logged)))
  }

  def fatal(msg: String, t: Truncatable, cause: Throwable): Nothing = {
    val (screen, logged) = t.strings
    throw new HailException(format(msg, screen), Some(format(msg, logged)), cause)
  }

  def fatal(msg: String, t1: Truncatable, t2: Truncatable): Nothing = {
    val (screen1, logged1) = t1.strings
    val (screen2, logged2) = t2.strings
    throw new HailException(format(msg, screen1, screen2), Some(format(msg, logged1, logged2)))
  }

  def deepestMessage(e: Throwable): String = {
    var iterE = e
    while (iterE.getCause != null)
      iterE = iterE.getCause

    s"${iterE.getClass.getSimpleName}: ${iterE.getMessage}"
  }

  def expandException(e: Throwable, logMessage: Boolean): String = {
    val msg = e match {
      case e: HailException => e.logMsg.filter(_ => logMessage).getOrElse(e.msg)
      case _ => e.getLocalizedMessage
    }
    s"${e.getClass.getName}: $msg\n\tat ${e.getStackTrace.mkString("\n\tat ")}\n\n${Option(
        e.getCause
      ).map(exception => expandException(exception, logMessage)).getOrElse("")}\n"
  }

  def handleForPython(e: Throwable): (String, String, Int) = {
    val short = deepestMessage(e)
    val expanded = expandException(e, logMessage = false)

    @tailrec def searchForErrorCode(exception: Throwable): Int =
      exception match {
        case e: HailException =>
          e.errorId
        case _ =>
          val cause = exception.getCause
          if (cause == null) -1 else searchForErrorCode(cause)
      }

    val error_id = searchForErrorCode(e)

    (short, expanded, error_id)
  }
}
