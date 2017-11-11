package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class AltAlleleView(variant: TAltAllele, source: TStruct) {
  private val view = new StructView(variant.representation, source)

  def setRegion(rv: RegionValue) {
    view.setRegion(rv)
  }

  def setRegion(m: MemoryBuffer, offset: Long) {
    view.setRegion(m, offset)
  }

  def hasRef(): Boolean = view.hasField("ref")
  def getRef(): String = view.getStringField("ref")
  def hasAlt(): Boolean = view.hasField("alt")
  def getAlt(): String = view.getStringField("alt")
}
