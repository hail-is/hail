package is.hail.shuffler

import is.hail.asm4s._
import org.apache.log4j.Logger

import scala.reflect.{ClassTag, classTag}

object LoggerCode {
  def getLogger[T: ClassTag](): Code[Logger] =
    Code.invokeStatic1[Logger, String, Logger]("getLogger", classTag[T].runtimeClass.getName())
}


class LoggerCode(
  code: Value[Logger]
) extends Value[Logger] {
  def get: Code[Logger] = code.get

  def info(message: Code[String]): Code[Unit] =
    code.invoke[java.lang.Object, Unit]("info", message)
  def warn(message: Code[String]): Code[Unit] =
    code.invoke[java.lang.Object, Unit]("warn", message)
  def error(message: Code[String]): Code[Unit] =
    code.invoke[java.lang.Object, Unit]("error", message)
}
