package is.hail.rvd

import is.hail.annotations.{RegionValue, WritableRegionValue}
import is.hail.expr.types.Type

/**
  * A buffered iterator which stabilizes the head so that region associated
  * with the underlying iterator may change without warning.
  */
class StableBufferedIterator(t: Type, child: Iterator[RegionValue]) extends BufferedIterator[RegionValue] {
  private[this] val hd = WritableRegionValue(t)
  private[this] var hdDefined: Boolean = false

  def head: RegionValue = {
    if (!hdDefined) {
      hd.set(next())
      hdDefined = true
    }
    hd.value
  }

  def hasNext =
    hdDefined || child.hasNext

  def next() =
    if (hdDefined) {
      hdDefined = false
      hd.value
    } else child.next()
}