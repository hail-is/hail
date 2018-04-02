package is.hail.rvd

import is.hail.annotations.Region
import scala.collection.mutable

object RVDContext {
  def default: RVDContext = new RVDContext(Region())

  def fromRegion(region: Region): RVDContext = new RVDContext(region)
}

// NB: must be *Auto*Closeable because calling close twice is undefined behavior
// (see AutoCloseable javadoc)
class RVDContext(r: Region) extends AutoCloseable {
  private[this] val children: mutable.ArrayBuffer[RVDContext] = new mutable.ArrayBuffer()

  private[this] def own(child: RVDContext): Unit = children += child

  def freshContext: RVDContext = {
    val ctx = RVDContext.default
    own(ctx)
    ctx
  }

  def region: Region = r // lifetime: element

  def partitionRegion: Region = r // lifetime: partition

  // frees the memory associated with this context
  def close(): Unit = ()
}
