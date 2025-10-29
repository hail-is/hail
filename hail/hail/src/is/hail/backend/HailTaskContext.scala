package is.hail.backend

import is.hail.annotations.RegionPool
import is.hail.utils._

import scala.collection.mutable

import java.io.Closeable

class TaskFinalizer {
  val closeables = mutable.ArrayBuffer.empty[Closeable]

  def clear(): Unit =
    closeables.clear()

  def addCloseable(c: Closeable): Unit =
    closeables += c

  def closeAll(): Unit = closeables.foreach(_.close())
}

abstract class HailTaskContext extends AutoCloseable {
  def stageId(): Int

  def partitionId(): Int

  def attemptNumber(): Int

  private lazy val thePool = RegionPool()

  def getRegionPool(): RegionPool = thePool

  def partSuffix(): String = {
    val rng = new java.security.SecureRandom()
    val fileUUID = new java.util.UUID(rng.nextLong(), rng.nextLong())
    s"${stageId()}-${partitionId()}-${attemptNumber()}-$fileUUID"
  }

  val finalizers = mutable.ArrayBuffer.empty[TaskFinalizer]

  def newFinalizer(): TaskFinalizer = {
    val f = new TaskFinalizer
    finalizers += f
    f
  }

  def close(): Unit = {
    log.info(
      s"TaskReport: stage=${stageId()}, partition=${partitionId()}, attempt=${attemptNumber()}, " +
        s"peakBytes=${thePool.getHighestTotalUsage}, peakBytesReadable=${formatSpace(thePool.getHighestTotalUsage)}, " +
        s"chunks requested=${thePool.getUsage._1}, cache hits=${thePool.getUsage._2}"
    )
    finalizers.foreach(_.closeAll())
    thePool.close()
  }
}
