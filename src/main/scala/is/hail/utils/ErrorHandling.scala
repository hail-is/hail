package is.hail.utils

trait ErrorHandling {
  def fatal(msg: String): Nothing = {
    throw new FatalException(msg)
  }

  def fatal(msg: String, t: Truncatable): Nothing = {
    val (screen, logged) = t.strings
    throw new FatalException(format(msg, screen), Some(format(msg, logged)))
  }

  def fatal(msg: String, t1: Truncatable, t2: Truncatable): Nothing = {
    val (screen1, logged1) = t1.strings
    val (screen2, logged2) = t2.strings
    throw new FatalException(format(msg, screen1, screen2), Some(format(msg, logged1, logged2)))
  }


  private def fail(msg: String): Nothing = {
    log.error(msg)
    System.err.println(msg)
    sys.exit(1)
  }

  def handleFatal(e: FatalException): Nothing = {
    log.error(s"hail: fatal: ${ e.logMsg }")
    System.err.println(s"hail: fatal: ${ e.msg }")
    sys.exit(1)
  }

  def digForFatal(e: Throwable): Option[String] = {
    val r = e match {
      case f: FatalException =>
        println(s"found fatal $f")
        Some(s"${ e.getMessage }")
      case _ =>
        Option(e.getCause).flatMap(c => digForFatal(c))
    }
    r
  }

  def deepestMessage(e: Throwable): String = {
    var iterE = e
    while (iterE.getCause != null)
      iterE = iterE.getCause

    s"${ iterE.getClass.getSimpleName }: ${ iterE.getLocalizedMessage }"
  }

  def expandException(e: Throwable): String = {
    val msg = e match {
      case f: FatalException => f.logMsg.getOrElse(f.msg)
      case _ => e.getLocalizedMessage
    }
    s"${ e.getClass.getName }: $msg\n\tat ${ e.getStackTrace.mkString("\n\tat ") }${
      Option(e.getCause).map(exception => expandException(exception)).getOrElse("")
    }"
  }

  def getMinimalMessage(e: Throwable): String = {
    val fatalOption = digForFatal(e)
    val prefix = if (fatalOption.isDefined) "fatal" else "caught exception"
    val msg = fatalOption.getOrElse(deepestMessage(e))
    log.error(s"hail: $prefix: $msg\nFrom ${ expandException(e) }")
    msg
  }
}
