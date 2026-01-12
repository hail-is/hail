package is.hail.utils

import is.hail.collection.{FastSeq, ObjectArrayStack}
import is.hail.utils.ExecutionTimer.{TimeBlock, Timings}

import scala.collection.mutable

import org.json4s.{JArray, JString, JValue}
import org.json4s.JsonAST.JLong
import sourcecode.Enclosing

object ExecutionTimer extends Logging {

  def time[T](f: ExecutionTimer => T)(implicit E: Enclosing): (T, Timings) = {
    val timer = new ExecutionTimer(E.value)
    val result = f(timer)
    timer.finish()
    timer.logInfo()
    (result, timer.result)
  }

  def logTime[T](f: ExecutionTimer => T)(implicit E: Enclosing): T = {
    val (result, _) = time[T](f)
    result
  }

  sealed trait Timings {
    def toJSON: JValue
  }

  private class TimeBlock(val name: String) extends Timings {
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

      logger.info(
        s"timing ${selfPrefix.mkString("/")} total ${formatTime(totalTime)} self ${formatTime(
            totalTime - childrenTime
          )} children ${formatTime(childrenTime)} %children ${formatDouble(childrenTime.toDouble * 100 / totalTime, 2)}%"
      )

      var i = 0
      while (i < children.length) {
        children(i).logInfo(selfPrefix)
        i += 1
      }
    }

    override def toJSON: JValue = {
      assert(finished)
      JArray(
        List(
          JString(name),
          JLong(totalTime),
          JLong(totalTime - childrenTime),
          JArray(children.map(_.toJSON).toList),
        )
      )
    }
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
      stack.pop(): Unit
    }
  }

  def finish(): Unit = {
    if (finished)
      return
    val end = System.nanoTime()
    rootBlock.finish(end - start)
    stack.pop(): Unit
    assert(stack.size == 0)
    finished = true
  }

  def logInfo(): Unit = {
    assert(finished)
    rootBlock.logInfo(FastSeq.empty[String])
  }

  def result: Timings = {
    assert(finished)
    rootBlock
  }
}
