package is.hail.backend

import is.hail.annotations.RegionPool
import is.hail.utils._

import java.io.Closeable

class TaskFinalizer {
  val closeables = new BoxedArrayBuilder[Closeable]()

  def clear(): Unit =
    closeables.clear()

  def addCloseable(c: Closeable): Unit =
    closeables += c

  def closeAll(): Unit =
    (0 until closeables.size).foreach(i => closeables(i).close())
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

  val finalizers = new BoxedArrayBuilder[TaskFinalizer]()

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
    (0 until finalizers.size).foreach(i => finalizers(i).closeAll())
    thePool.close()
  }
}
