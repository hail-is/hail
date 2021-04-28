package is.hail.backend

import is.hail.annotations.RegionPool
import is.hail.utils._

abstract class HailTaskContext {
  def stageId(): Int

  def partitionId(): Int

  def attemptNumber(): Int

  private lazy val thePool = RegionPool()

  def getRegionPool(): RegionPool = thePool

  def partSuffix(): String = {
    val rng = new java.security.SecureRandom()
    val fileUUID = new java.util.UUID(rng.nextLong(), rng.nextLong())
    s"${ stageId() }-${ partitionId() }-${ attemptNumber() }-$fileUUID"
  }

  def finish(): Unit = {
    log.info(s"TaskReport: stage=${ stageId() }, partition=${ partitionId() }, attempt=${ attemptNumber() }, " +
      s"peakBytes=${ thePool.getHighestTotalUsage }, peakBytesReadable=${ formatSpace(thePool.getHighestTotalUsage) }")
    thePool.close()
  }
}
