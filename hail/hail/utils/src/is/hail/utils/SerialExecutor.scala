package is.hail.utils

import java.io.Closeable
import java.util.concurrent.{
  CancellationException, ExecutionException, Executors, FutureTask, TimeUnit,
}
import java.util.concurrent.atomic.AtomicReference

// Runs operations serially on one permanent daemon thread. At most one
// operation is ever in flight: concurrent submissions are rejected rather
// than queued behind it. The in-flight operation can be interrupted via
// `cancel`.
final class SerialExecutor(threadName: String) extends Closeable {

  private[this] val inFlight =
    new AtomicReference[FutureTask[_]]()

  private[this] val executor =
    Executors.newSingleThreadExecutor { r =>
      val t = new Thread(r, threadName)
      t.setDaemon(true)
      t
    }

  def run[T](f: => T): T = {
    val task = new FutureTask[T](() => f)
    if (!inFlight.compareAndSet(null, task))
      fatal("another operation is in progress; wait for it to finish or cancel it")
    try {
      executor.execute(task)
      task.get()
    } catch {
      case e: ExecutionException => throw e.getCause
      case _: CancellationException => fatal("operation was cancelled")
    } finally inFlight.compareAndSet(task, null): Unit
  }

  def cancel(): Unit =
    Option(inFlight.get()).foreach(_.cancel(true))

  // await in-flight work so it unwinds before callers tear down the state it
  // uses; termination also publishes the worker thread's writes to us
  override def close(): Unit = {
    cancel()
    executor.shutdown()
    if (!executor.awaitTermination(5, TimeUnit.SECONDS))
      fatal(
        "An operation is still running and did not respond to interrupt.\n" +
          "You should restart the hail session."
      )
  }
}
