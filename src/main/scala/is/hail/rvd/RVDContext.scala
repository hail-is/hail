package is.hail.rvd

import is.hail.annotations.Region
import scala.collection.mutable

object RVDContext {
  def default: RVDContext = new RVDContext(Region())

  def fromRegion(region: Region): RVDContext = new RVDContext(region)
}

// NB: must be *Auto*Closeable because calling close twice is undefined behavior
// (see AutoCloseable javadoc)
class RVDContext(var active: Region) extends AutoCloseable {
  private[this] val regions: mutable.ArrayBuffer[Region] = new mutable.ArrayBuffer()
  regions += active

  // this method could use a re-think, but the idea is thus:
  //
  // There must be a unique context object for every partition (the context
  // object corresponds exactly to the lifetime of a partition, from read to
  // serialization (whether serialized and sent to master or to another worker).
  //
  // Calling freshRegion mutates the context object. It should be called *after*
  // the producer has captured references to the old
  // region. `ContextRDD.cmapPartitions` is designed to provide the context to
  // the producers first. Users of `ContextRDD` are designed to capture the
  // provided region. When control flows to the consumer, the consumer may call
  // freshRegion, which points the context at a new active region. The
  // producer's iterators will still reference the old active region.
  //
  // In turn, the consumer becomes a producer. Its consmers will see the
  // freshRegion as the active region. They may call freshRegion again,
  // continuing the cycle.
  def freshRegion(): Region = {
    val old = active
    active = Region()
    println(s"creating a freshRegion: $active was $old")
    regions += active
    old
  }

  def region: Region = active // lifetime: element

  def partitionRegion: Region = active // lifetime: partition

  // frees the memory associated with this context
  def close(): Unit = ()
}
