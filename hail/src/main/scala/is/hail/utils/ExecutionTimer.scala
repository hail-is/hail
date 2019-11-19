package is.hail.utils

import scala.collection.mutable


case class TimeBlock(name: String, parent: Option[TimeBlock]) {
  parent.foreach(_.isLeaf = false)
  private val start: Long = System.nanoTime()
  private var end: Long = 0L
  private var finished: Boolean = false
  private var isLeaf: Boolean = true
  private var coveredDuration: Long = 0L

  def isLeafNode: Boolean = isLeaf

  def finish(): Unit = {
    assert(!finished)
    end = System.nanoTime()
    finished = true
    parent.foreach { p =>
      assert(!p.isLeaf)
      p.coveredDuration += getCoveredDuration
    }
  }

  def getCoveredDuration: Long = {
    assert(finished)
    if (isLeaf)
      end - start
    else
      coveredDuration
  }

  private def appendPath(ab: ArrayBuilder[String]): Unit = {
    parent.foreach(_.appendPath(ab))
    ab += name
  }

  def path(): Array[String] = {
    val ab = new ArrayBuilder[String]
    appendPath(ab)
    ab.result()
  }

  def duration: Long = {
    assert(finished)
    end - start
  }

  def fmtDuration: String = {
    formatTime(duration)
  }

  def report(jobStartTime: Long): Unit = {
    assert(finished)
    val cov = if (!isLeaf) s", tagged coverage ${ formatDouble(coveredDuration.toDouble / duration * 100, 1) }" else ""
    log.info(s"Timer: Time taken for ${ path().mkString(" -- ") } : $fmtDuration, total ${ formatTime(end - jobStartTime) }$cov")
  }
}

class ExecutionTimer {
  private val stack = new ArrayStack[TimeBlock]()
  private val measurements: mutable.ArrayBuffer[TimeBlock] = mutable.ArrayBuffer.empty

  def time[T](tag: String)(block: => T): T = {
    assert(!finished)
    val preSize = stack.size
    stack.push(TimeBlock(tag, stack.topOption))
    val result = block
    val finishedBlock = stack.pop()
    finishedBlock.finish()
    finishedBlock.report(startTime)
    measurements += finishedBlock
    assert(stack.size == preSize)
    result
  }

  private val startTime = System.nanoTime()

  private var finished: Boolean = false
  private var endTime: Long = _

  def register(block: TimeBlock): Unit = {
    assert(!finished)
    block.report(startTime)
    measurements += block
  }

  def finish(): Unit = {
    finished = true
    endTime = System.nanoTime()
  }

  def logInfo() {
    assert(finished)
    log.info("Timer: all timings:")
    measurements.foreach { t =>
      t.report(startTime)
    }
    log.info("Timer: aggregate:")
    measurements.filter { t => t.isLeafNode }
      .groupBy(_.name)
      .toArray
      .map { case (tag, ab) => (tag, ab.map(_.duration).sum, ab.size) }
      .sortBy(_._2)
      .foreach { case (tag, sum, n) =>
        log.info(s"Timer: Time taken for tag '$tag' ($n): ${ formatTime(sum) }")
      }
    val taggedSum = measurements.iterator.filter { t => t.isLeafNode }.map(_.duration).sum
    val totalTime = endTime - startTime
    val percentCovered = taggedSum.toDouble / totalTime * 100
    log.info(s"Timer: Fraction covered by a tagged leaf: ${ formatTime(totalTime - taggedSum) } (${ formatDouble(percentCovered, 1) }%)")
  }

  def asMap(): mutable.Map[String, Long] = {
    assert(finished)
    val m = mutable.Map.empty[String, Long]
    measurements.foreach { t =>
      m += ((t.path().mkString(" -- "), t.duration))
    }
    m
  }
}
