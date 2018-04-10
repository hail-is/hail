package is.hail.rvd

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.sparkextras.ResettableContext

import scala.collection.mutable

object RVDContext {
  def default: RVDContext = new RVDContext(Region())

  def fromRegion(region: Region): RVDContext = new RVDContext(region)
}

class RVDContext(r: Region) extends ResettableContext {
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

  private[this] val theRvb = new RegionValueBuilder(r)
  def rvb = theRvb

  def reset(): Unit = {
    r.clear()
  }

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
