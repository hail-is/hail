package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class RegionValueAltAllele(taa: TAltAllele) extends View with AltAllele {
  private val t = taa.fundamentalType.asInstanceOf[TStruct]
  private val refIdx = t.fieldIdx("ref")
  private val altIdx = t.fieldIdx("alt")
  private var region: MemoryBuffer = _
  private var offset: Long = _
  private var _ref: String = null
  private var _alt: String = null

  assert(t.isFieldRequired(refIdx))
  assert(t.isFieldRequired(altIdx))

  def setRegion(region: MemoryBuffer, offset: Long) {
    this.region = region
    this.offset = offset
    this._ref = null
    this._alt = null
  }

  def ref(): String = {
    if (_ref == null)
      _ref = TString.loadString(region, t.loadField(region, offset, refIdx))
    _ref
  }

  def alt(): String = {
    if (_alt == null)
      _alt = TString.loadString(region, t.loadField(region, offset, altIdx))
    _alt
  }
}
