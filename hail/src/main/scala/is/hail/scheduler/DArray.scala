package is.hail.scheduler

abstract class DArray[T] {
  type Context

  val contexts: Array[Context]
  val body: Context => T

  def nTasks: Int = contexts.length
}
