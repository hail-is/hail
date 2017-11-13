package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class AltAlleleView(altAllele: TAltAllele, source: TStruct) {
  private val view = new StructView(altAllele.representation, source)

  def setRegion(rv: RegionValue) {
    view.setRegion(rv)
  }

  def setRegion(m: MemoryBuffer, offset: Long) {
    view.setRegion(m, offset)
  }

  private val refId = altAllele.representation.fieldIdx("ref")
  def hasRef(): Boolean = view.hasField(refId)
  def getRef(): String = view.getStringField(refId)
  private val altId = altAllele.representation.fieldIdx("alt")
  def hasAlt(): Boolean = view.hasField(altId)
  def getAlt(): String = view.getStringField(altId)
}
