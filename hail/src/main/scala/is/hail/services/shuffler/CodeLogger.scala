package is.hail.services.shuffler

import is.hail.asm4s._
import org.apache.log4j.Logger

import scala.reflect.{ClassTag, classTag}

object CodeLogger {
  def getLogger[T: ClassTag](): Code[Logger] =
    Code.invokeStatic1[Logger, String, Logger]("getLogger", classTag[T].runtimeClass.getName())
}
