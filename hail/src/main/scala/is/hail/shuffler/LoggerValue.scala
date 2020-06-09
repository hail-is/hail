package is.hail.shuffler

import is.hail.asm4s._
import org.apache.log4j.Logger

class LoggerValue (
  val code: Value[Logger]
) extends AnyVal {
  def info(message: Code[String]): Code[Unit] =
    code.invoke[java.lang.Object, Unit]("info", message)
  def warn(message: Code[String]): Code[Unit] =
    code.invoke[java.lang.Object, Unit]("warn", message)
  def error(message: Code[String]): Code[Unit] =
    code.invoke[java.lang.Object, Unit]("error", message)
}
