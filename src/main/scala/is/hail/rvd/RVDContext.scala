package is.hail.rvd

import is.hail.annotations.Region

object RVDContext {
  def default: RVDContext = new RVDContext(Region())

  def fromRegion(region: Region): RVDContext = new RVDContext(region)
}

// NB: must be *Auto*Closeable because calling close twice is undefined behavior
// (see AutoCloseable javadoc)
class RVDContext(r: Region) extends AutoCloseable {
  def region: Region = r // lifetime: element

  def partitionRegion: Region = r // lifetime: partition

  // frees the memory associated with this context
  def close(): Unit = ()
}
