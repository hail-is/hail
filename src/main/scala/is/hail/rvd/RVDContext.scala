package is.hail.rvd

import is.hail.annotations.{Region, RegionValueBuilder}

import scala.collection.mutable

object RVDContext {
  def default: RVDContext = new RVDContext(Region())

  def fromRegion(region: Region): RVDContext = new RVDContext(region)
}

class RVDContext(r: Region) extends AutoCloseable {
  private[this] val children = new mutable.ArrayBuffer[AutoCloseable]()
  private[this] val earlyClosedChildren = new mutable.HashSet[AutoCloseable]()

  private[this] def own(child: AutoCloseable): Unit = children += child

  own(r)

  def freshContext: RVDContext = {
    val ctx = RVDContext.default
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
    var i = 0
    while (i < children.size) {
      if (!earlyClosedChildren.contains(children(i))) {
        try {
          children(i).close()
        } catch {
          case e2: Exception =>
            if (e == null)
              e = e2
            else
              e.addSuppressed(e2)
        }
      }
      i += 1
    }

    if (e != null)
      throw e
  }

  def closeChild(child: AutoCloseable): Unit = {
    assert(children.contains(child)) // FIXME: remove for performance reasons
    child.close()
    earlyClosedChildren += child
  }
}
