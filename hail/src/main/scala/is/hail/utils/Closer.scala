package is.hail.utils

import org.apache.spark.TaskContext

final class Closer(x: AutoCloseable) extends (TaskContext => Unit) {
  def apply(taskContext: TaskContext): Unit = {
    x.close()
  }
}
