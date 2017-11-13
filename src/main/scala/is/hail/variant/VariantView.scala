package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class VariantView(variant: TVariant, source: TStruct) {
  private val view = new StructView(variant.representation, source)

  def setRegion(rv: RegionValue) {
    view.setRegion(rv)
  }

  def setRegion(m: MemoryBuffer, offset: Long) {
    view.setRegion(m, offset)
  }

  private val contigIdx = variant.representation.fieldIdx("contig")
  def hasContig(): Boolean = view.hasField(contigIdx)
  def getContig(): String = view.getStringField(contigIdx)
  private val startIdx = variant.representation.fieldIdx("start")
  def hasStart(): Boolean = view.hasField(startIdx)
  def getStart(): Int = view.getIntField(startIdx)
  private val refIdx = variant.representation.fieldIdx("ref")
  def hasRef(): Boolean = view.hasField(refIdx)
  def getRef(): String = view.getStringField(refIdx)
  private val altAllelesIdx = variant.representation.fieldIdx("altAlleles")
  def hasAltAlleles(): Boolean = view.hasField(altAllelesIdx)
  def getAltAlleles(): Long = view.getArrayField(altAllelesIdx)
}
