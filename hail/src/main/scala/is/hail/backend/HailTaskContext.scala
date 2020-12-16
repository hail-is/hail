package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.RegionPool
import is.hail.backend.local.{LocalBackend, LocalTaskContext}
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import org.apache.spark.TaskContext

object HailTaskContext {
  def get(): HailTaskContext = taskContext.get

  private[this] val taskContext: ThreadLocal[HailTaskContext] = new ThreadLocal[HailTaskContext]() {
    override def initialValue(): HailTaskContext = {
        val sparkTC = TaskContext.get()
        assert(sparkTC != null, "Spark Task Context was null, maybe this ran on the driver?")
        new SparkTaskContext(sparkTC)
    }
  }
  def setTaskContext(tc: HailTaskContext): Unit = taskContext.set(tc)
  def finish(): Unit = {
    taskContext.get().getRegionPool().close()
    taskContext.remove()
  }
}

abstract class HailTaskContext {
  type BackendType
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
}