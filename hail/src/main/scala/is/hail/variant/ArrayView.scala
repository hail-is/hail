package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types.TArray
import is.hail.utils._

class ArrayView[V <: View](t: TArray, val elementView: V) {
  private var region: Region = _
  private var aoff: Long = _
  private var eoff: Long = _
  private var _length: Int = _

  def setRegion(region: Region, aoff: Long) {
    this.region = region
    this.aoff = aoff
    this._length = t.loadLength(region, aoff)
  }

  def set(i: Int) {
    this.eoff = t.loadElement(region, aoff, length, i)
    elementView.setRegion(region, eoff)
  }

  def length(): Int = _length
}
