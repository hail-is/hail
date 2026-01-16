package is.hail.sparkextras.implicits

import org.apache.spark.TaskContext

class RichTaskContext(val ctx: TaskContext) extends AnyVal {
  def partSuffix: String = {
    val rng = new java.security.SecureRandom()
    val fileUUID = new java.util.UUID(rng.nextLong(), rng.nextLong())
    s"${ctx.stageId()}-${ctx.partitionId()}-${ctx.attemptNumber()}-$fileUUID"
  }
}
