package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class VariantView(variant: TVariant, source: TStruct) {
  private val view = new StructView(variant.representation, source)

  def setRegion(m: MemoryBuffer, offset: Long) {
    view.setRegion(m, offset)
  }

  def hasContig(): Boolean = view.hasField("contig")
  def getContig(): String = view.getStringField("contig")
  def hasStart(): Boolean = view.hasField("start")
  def getStart(): Int = view.getIntField("start")
  def hasRef(): Boolean = view.hasField("ref")
  def getRef(): String = view.getStringField("ref")
  def hasAltAlleles(): Boolean = view.hasField("altAlleles")
  def getAltAlleles(): Long = view.getArrayField("altAlleles")
}
