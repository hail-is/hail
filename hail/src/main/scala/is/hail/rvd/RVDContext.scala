package is.hail.rvd

import is.hail.annotations.{Region, RegionValueBuilder}

import scala.collection.mutable

object RVDContext {
  def default: RVDContext = {
    val partRegion = Region()
    val ctx = new RVDContext(Region(), Region())
    ctx.own(partRegion)
    ctx
  }
}

class RVDContext(val partitionRegion: Region, val r: Region) extends AutoCloseable {
  private[this] val children = new mutable.HashSet[AutoCloseable]()

  private def own(child: AutoCloseable): Unit = children += child
  private[this] def disown(child: AutoCloseable): Unit =
    assert(children.remove(child))

  own(r)

  def freshContext: RVDContext = {
    val ctx = new RVDContext(partitionRegion, Region())
    own(ctx)
    ctx
  }

  def freshRegion: Region = {
    val r2 = Region()
    own(r2)
    r2
  }

  def region: Region = r

  private[this] val theRvb = new RegionValueBuilder(r)
  def rvb = theRvb

  // frees the memory associated with this context
  def close(): Unit = {
    var e: Exception = null
    children.foreach { child =>
      try {
        child.close()
      } catch {
        case e2: Exception =>
          if (e == null)
            e = e2
          else
            e.addSuppressed(e2)
      }
    }

    if (e != null)
      throw e
  }

  def closeChild(child: AutoCloseable): Unit = {
    child.close()
    disown(child)
  }
}
