package is.hail.backend

import is.hail.annotations.{Region, RegionPool}
import is.hail.utils.using

import scala.collection.mutable

trait HailTaskContext {

  /** region whose lifetime is at least as long as this task */
  def r: Region

  /** register an action that will be called when this task completes */
  def onClose(f: () => Unit): Unit
}

object HailTaskContext {
  def runPartition[A](partId: Int)(f: HailTaskContext => A): A =
    using(new PartitionContext(partId))(f)
}

class PartitionContext(partId: Int) extends HailTaskContext with AutoCloseable {
  private[this] val onCloseTasks = mutable.ArrayBuffer.empty[() => Unit]

  private[this] val pool = RegionPool()
  override val r: Region = Region(pool = pool)
  override def onClose(f: () => Unit): Unit = onCloseTasks += f

  override def close(): Unit = {
    onCloseTasks.foreach(_())
    r.close()
    pool.logStats(s"Partition $partId")
    pool.close()
  }
}
