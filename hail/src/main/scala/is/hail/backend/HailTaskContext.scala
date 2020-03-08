package is.hail.backend

object HailTaskContext {
  def get(): HailTaskContext = taskContext.get

  private[this] val taskContext: ThreadLocal[HailTaskContext] = new ThreadLocal[HailTaskContext]
  protected[backend] def setTaskContext(tc: HailTaskContext): Unit = taskContext.set(tc)
  protected[backend] def unset(): Unit = taskContext.remove()
}

abstract class HailTaskContext {
  type BackendType
  def stageId(): Int
  def partitionId(): Int
  def attemptNumber(): Int

  def partSuffix(): String = {
    val rng = new java.security.SecureRandom()
    val fileUUID = new java.util.UUID(rng.nextLong(), rng.nextLong())
    s"${ stageId() }-${ partitionId() }-${ attemptNumber() }-$fileUUID"
  }
}