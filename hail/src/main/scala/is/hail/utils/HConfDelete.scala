package is.hail.utils

import org.apache.spark.TaskContext
import org.apache.hadoop.conf.Configuration

class HConfDelete(hConf: Configuration, partPath: String, recursive: Boolean) extends (TaskContext => Unit) {
  def apply(taskContext: TaskContext): Unit = {
    hConf.delete(partPath, recursive)
  }
}
