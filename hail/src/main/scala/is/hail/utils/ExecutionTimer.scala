package is.hail.utils

import scala.collection.mutable

import sourcecode.Enclosing

class TimeBlock(val name: String) {
  val children: mutable.ArrayBuffer[TimeBlock] = new mutable.ArrayBuffer()
  var totalTime: Long = 0L
  var childrenTime: Long = 0L
  var finished: Boolean = false

  def finish(t: Long): Unit = {
    assert(!finished)
    finished = true

    totalTime = t

    var i = 0
    while (i < children.length) {
      childrenTime += children(i).totalTime
      i += 1
    }
  }

  def logInfo(prefix: IndexedSeq[String]): Unit = {
    assert(finished)

    val selfPrefix = prefix :+ name

    log.info(s"timing ${selfPrefix.mkString("/")} total ${formatTime(totalTime)} self ${formatTime(
        totalTime - childrenTime
      )} children ${formatTime(childrenTime)} %children ${formatDouble(childrenTime.toDouble * 100 / totalTime, 2)}%")

    var i = 0
    while (i < children.length) {
      children(i).logInfo(selfPrefix)
      i += 1
    }
  }

  def toMap: Map[String, Any] = {
    assert(finished)
    Map[String, Any](
      "name" -> name,
      "total_time" -> totalTime,
      "self_time" -> (totalTime - childrenTime),
      "children_time" -> childrenTime,
      "children" -> children.map(_.toMap),
    )
  }
}

object ExecutionTimer {
  def time[T](f: ExecutionTimer => T)(implicit E: Enclosing): (T, Map[String, Any]) = {
    val timer = new ExecutionTimer(E.value)
    val result = f(timer)
    timer.finish()
    timer.logInfo()
    (result, timer.toMap)
  }

  def logTime[T](f: ExecutionTimer => T)(implicit E: Enclosing): T = {
    val (result, _) = time[T](f)
    result
  }
}

class ExecutionTimer(val rootName: String) {
  private[this] val stack = new ObjectArrayStack[TimeBlock]()

  private[this] val rootBlock = new TimeBlock(rootName)
  stack.push(rootBlock)

  private[this] val start: Long = System.nanoTime()

  private[this] var finished: Boolean = false

  def time[T](name: String)(block: => T): T = {
    assert(!finished)
    val parent = stack.top
    val child = new TimeBlock(name)
    parent.children += child
    stack.push(child)
    val start = System.nanoTime()
    try block
    finally {
      val end = System.nanoTime()
      child.finish(end - start)
      stack.pop()
    }
  }

  def finish(): Unit = {
    if (finished)
      return
    val end = System.nanoTime()
    rootBlock.finish(end - start)
    stack.pop()
    assert(stack.size == 0)
    finished = true
  }

  def logInfo(): Unit = {
    assert(finished)
    rootBlock.logInfo(FastSeq.empty[String])
  }

  def toMap: Map[String, Any] = {
    assert(finished)
    rootBlock.toMap
  }
}
