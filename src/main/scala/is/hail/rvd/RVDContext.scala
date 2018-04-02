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
  private[this] val children: mutable.ArrayBuffer[AutoCloseable] = new mutable.ArrayBuffer()

  private[this] def own(child: AutoCloseable): Unit = children += child

  own(r)

  def freshContext: RVDContext = {
    val ctx = RVDContext.default
    own(ctx)
    ctx
  }

  def region: Region = r // lifetime: element

  def partitionRegion: Region = r // lifetime: partition

  // frees the memory associated with this context
  def close(): Unit = {
    var e: Exception = null
    var i = 0
    while (i < children.size) {
      try {
        children(i).close()
      } catch {
        case e2: Exception =>
          if (e == null)
            e = e2
          else
            e.addSuppressed(e2)
      }
      i += 1
    }

    if (e != null)
      throw e
  }
}
