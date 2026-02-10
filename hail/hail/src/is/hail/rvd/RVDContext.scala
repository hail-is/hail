package is.hail.rvd

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.backend.{HailStateManager, HailTaskContext}

import scala.collection.mutable

object RVDContext {
  def default(tc: HailTaskContext) =
    new RVDContext(tc.r, Region(pool = tc.r.pool))
}

class RVDContext(override val r: Region, val region: Region)
    extends HailTaskContext with AutoCloseable {
  private[this] val children = mutable.HashSet.empty[AutoCloseable]

  private def own(child: AutoCloseable): Unit = children += child

  private[this] def disown(child: AutoCloseable): Unit =
    assert(children.remove(child))

  own(region)

  def freshContext(): RVDContext = {
    val ctx = RVDContext.default(this)
    own(ctx)
    ctx
  }

  def freshRegion(blockSize: Region.Size = Region.REGULAR): Region = {
    val r2 = Region(blockSize, pool = r.pool)
    own(r2)
    r2
  }

  override def onClose(f: () => Unit): Unit =
    own(() => f())

  private[this] val theRvb = new RegionValueBuilder(HailStateManager(Map.empty), region)
  def rvb = theRvb

  // frees the memory associated with this context
  override def close(): Unit = {
    var e: Exception = null
    children.foreach { child =>
      try
        child.close()
      catch {
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
