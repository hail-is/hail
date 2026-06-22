package is.hail.utils

import jdk.jfr.{Category, Event, Label, Name, StackTrace}
import sourcecode.Enclosing

@Category(Array("Hail"))
@Name("is.hail.utils.TimedBlock")
@Label("Timed Block")
@StackTrace(false)
class TimedBlock(@Label("Block Name") val name: String) extends Event

object TimedBlock {
  @inline def enter[A](f: => A)(implicit E: Enclosing): A =
    enter(E.value)(f)

  @inline def enter[A](context: String)(f: => A): A = {
    val event = new TimedBlock(context)
    if (!event.isEnabled) f
    else {
      event.begin()
      try f
      finally event.commit()
    }
  }
}
