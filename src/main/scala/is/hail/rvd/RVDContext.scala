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
  private[this] val regions: mutable.ArrayBuffer[Region] = new mutable.ArrayBuffer()
  regions += r

  def freshRegion(): Region = {
    val r2 = Region()
    regions += r2
    r2
  }

  def region: Region = r // lifetime: element

  def partitionRegion: Region = r // lifetime: partition

  // frees the memory associated with this context
  def close(): Unit = ()
}
