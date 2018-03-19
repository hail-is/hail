package is.hail.rvd

import is.hail.annotations.Region

object RVDContext {
  def default: RVDContext = SimpleRVDContext(Region())

  def fromRegion(region: Region): RVDContext = SimpleRVDContext(region)
}

// NB: must be *Auto*Closeable because calling close twice is undefined behavior
// (see AutoCloseable javadoc)
trait RVDContext extends AutoCloseable {
  def region: Region // lifetime: element

  def partitionRegion: Region // lifetime: partition

  // frees the memory associated with this context
  def close: Unit
}

case class SimpleRVDContext(region: Region) extends RVDContext {
  val partitionRegion = region

  def close {
    region.close()
  }
}
