package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class RegionValueAltAllele(taa: TAltAllele) extends View with IAltAllele {
  private val t = taa.fundamentalType.asInstanceOf[TStruct]
  private val refIdx = t.fieldIdx("ref")
  private val altIdx = t.fieldIdx("alt")
  private var region: MemoryBuffer = _
  private var offset: Long = _
  private var cachedRef: String = null
  private var cachedAlt: String = null

  assert(t.isFieldRequired(refIdx))
  assert(t.isFieldRequired(altIdx))

  def setRegion(region: MemoryBuffer, offset: Long) {
    this.region = region
    this.offset = offset
    this.cachedRef = null
    this.cachedAlt = null
  }

  def getOffset(): Long = offset

  def ref(): String = {
    if (cachedRef == null)
      cachedRef = TString.loadString(region, t.loadField(region, offset, refIdx))
    cachedRef
  }

  def alt(): String = {
    if (cachedAlt == null)
      cachedAlt = TString.loadString(region, t.loadField(region, offset, altIdx))
    cachedAlt
  }
}
