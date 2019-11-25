package is.hail.shuffler

import is.hail.HailContext
import is.hail.utils._
import java.util.concurrent.{ Callable, Executors, Future }
import org.apache.spark.TaskContext

object ShuffleAsyncPool {
  val pool = ThreadLocal.withInitial(() =>
    new AsyncPool(HailContext.get.flags.get("shuffle_read_parallelism").toInt))
}

class AsyncPool(
  private[this] val maxAsync: Int
) {
  private[this] val executor = Executors.newFixedThreadPool(maxAsync)
  TaskContext.get.addTaskCompletionListener(_ => executor.shutdownNow())

  def future[T](f: () => T): Future[T] = executor.submit(f)
}
