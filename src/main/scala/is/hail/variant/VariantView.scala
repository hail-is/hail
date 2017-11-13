package is.hail.variant

import is.hail.annotations._
import is.hail.expr._

class VariantView(t: TStruct) {
  private var region: MemoryBuffer = _
  private var offset: Long = _
  private val contigIdx: Int = t.fieldIdx("contig")
  private val startIdx: Int = t.fieldIdx("start")
  private val refIdx: Int = t.fieldIdx("ref")
  private val altAllelesIdx: Int = t.fieldIdx("altAlleles")
  private val tAltAlleles: TArray = t.fieldType(altAllelesIdx).asInstanceOf[TArray]
  private val tAltAllele: TStruct = tAltAlleles.elementType.asInstanceOf[TStruct]
  private val altIdx: Int = tAltAllele.fieldIdx("alt")

  // assert(t.fieldType(contigIdx).required)
  // assert(t.fieldType(startIdx).required)
  // assert(t.fieldType(refIdx).required)
  // assert(t.fieldType(altAllelesIdx).required)
  // assert(tAltAllele.fieldType(altIdx).required)

  def setRegion(rv: RegionValue) {
    region = rv.region
    offset = rv.offset
  }

  def setRegion(region: MemoryBuffer, offset: Long) {
    this.region = region
    this.offset = offset
  }

  def getContig(): String = {
    assert(t.isFieldDefined(region, offset, contigIdx))
    TString.loadString(region, t.loadField(region, offset, contigIdx))
  }

  def getStart(): Int = {
    assert(t.isFieldDefined(region, offset, startIdx))
    region.loadInt(t.loadField(region, offset, startIdx))
  }

  def getRef(): String = {
    assert(t.isFieldDefined(region, offset, refIdx))
    TString.loadString(region, t.loadField(region, offset, refIdx))
  }

  def getAltAlleles(): Long = {
    assert(t.isFieldDefined(region, offset, altAllelesIdx))
    t.loadField(region, offset, altAllelesIdx)
  }

  def getAlt(): String = {
    val aoff = getAltAlleles()
    assert(tAltAlleles.loadLength(region, aoff) == 1)
    assert(tAltAlleles.isElementDefined(region, aoff, 0))
    val eoff = tAltAlleles.loadElement(region, aoff, 1, 0)
    assert(tAltAllele.isFieldDefined(region, eoff, altIdx))
    TString.loadString(region, tAltAllele.loadField(region, eoff, altIdx))
  }
}
