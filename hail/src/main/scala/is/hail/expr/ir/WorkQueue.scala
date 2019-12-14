package is.hail.expr.ir

import java.util

import is.hail.utils.ArrayBuilder

trait Task[C] {
  def consume(ctx: C): Boolean
  def queueLeftovers(foo: ArrayBuilder[Task[C]]): Unit = ()
}

class WorkQueue[C](context: C) {
  private val queue = new util.ArrayDeque[Task[C]]()
  private val q2 = new ArrayBuilder[Task[C]]()

  def add(task: Task[C]): Unit = queue.addLast(task)
  def consume(): Unit = {
    val task = queue.pollFirst()
    val remaining = task.consume(context)
    if (remaining) {
      q2.clear()
      task.queueLeftovers(q2)
      while (q2.size > 0) {
        queue.addFirst(q2.pop())
      }
    }
  }
}
