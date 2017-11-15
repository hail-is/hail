package is.hail.variant

import is.hail.annotations._
import is.hail.expr._

class VariantView(tv: TVariant) extends View {
  private val t: TStruct = tv.representation.asInstanceOf[TStruct]
  private var region: MemoryBuffer = _
  private var offset: Long = _
  private val contigIdx: Int = t.fieldIdx("contig")
  private val startIdx: Int = t.fieldIdx("start")
  private val refIdx: Int = t.fieldIdx("ref")
  private val altAllelesIdx: Int = t.fieldIdx("altAlleles")
  private val tAltAlleles: TArray = t.fieldType(altAllelesIdx).asInstanceOf[TArray]
  private val tAltAllele: TAltAllele = tAltAlleles.elementType.asInstanceOf[TAltAllele]

  assert(t.fieldType(contigIdx).required)
  assert(t.fieldType(startIdx).required)
  assert(t.fieldType(refIdx).required)
  assert(tAltAlleles.required)
  assert(tAltAlleles.elementType.required)

  def setRegion(region: MemoryBuffer, offset: Long) {
    this.region = region
    this.offset = offset
  }

  def getContig(): String = {
    TString.loadString(region, t.loadField(region, offset, contigIdx))
  }

  def getStart(): Int = {
    region.loadInt(t.loadField(region, offset, startIdx))
  }

  def getRef(): String = {
    TString.loadString(region, t.loadField(region, offset, refIdx))
  }

  val altAlleles = new ArrayView(tAltAlleles, new AltAlleleView(tAltAllele))
  def loadAltAlleles() {
    altAlleles.setRegion(region, t.loadField(region, offset, altAllelesIdx))
  }

  // loadAltAlleles() must be called first
  def getAlt(): String = {
    assert(altAlleles.getLength() == 1)
    altAlleles.set(0)
    altAlleles.elementView.getAlt()
  }
}
