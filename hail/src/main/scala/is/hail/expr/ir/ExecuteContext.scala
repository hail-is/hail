package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.Region
import scala.collection.mutable.ArrayBuffer

object ExecuteContext {
  def scoped[T](f: ExecuteContext => T): T = {
    Region.scoped { r =>
      using(ExecuteContext(r,  new ExecutionTimer))(f)
    }
  }
}

case class ExecuteContext(r: Region, timer: ExecutionTimer) extends AutoCloseable {
  private[this] val onExits = new ArrayBuffer[() => Unit]()
  def addOnExit(onExit: () => Unit): Unit = {
    onExits += onExit
  }

  def close(): Unit = {
    onExits.foreach(_())
  }
}
