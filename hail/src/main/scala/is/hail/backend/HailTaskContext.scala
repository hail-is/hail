package is.hail.backend

import is.hail.annotations.RegionPool
import is.hail.backend.spark.SparkTaskContext
import org.apache.spark.TaskContext

object HailTaskContext {
  def get(): HailTaskContext = taskContext.get

  private[this] val taskContext: ThreadLocal[HailTaskContext] = new ThreadLocal[HailTaskContext]() {
    /**
      * initialValue should only be called when the task context runs on a Spark worker.
      * All driver runtimes and lowered (service, local) remote executions should set
      * the context manually.
      */
    override def initialValue(): HailTaskContext = {
      val sparkTC = TaskContext.get()
      assert(sparkTC != null, "Spark Task Context was null, maybe this ran on the driver?")
      TaskContext.get().addTaskCompletionListener[Unit] { (_: TaskContext) =>
        HailTaskContext.finish()
      }
      val htc = new SparkTaskContext(sparkTC)
      htc
    }
  }

  def setTaskContext(tc: HailTaskContext): Unit = taskContext.set(tc)

  def finish(): Unit = {
    taskContext.get().getRegionPool().close()
    taskContext.remove()
  }
}

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
}

class DriverTaskContext extends HailTaskContext {
  val stageId: Int = 0
  val partitionId: Int = 0
  val attemptNumber: Int = 0
}