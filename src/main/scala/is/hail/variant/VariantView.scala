package is.hail.variant

import is.hail.annotations._
import is.hail.expr._

class VariantView(tv: TVariant) extends Variant {
  private val t: TStruct = tv.fundamentalType.asInstanceOf[TStruct]
  private var region: MemoryBuffer = _
  private var offset: Long = _
  private val contigIdx: Int = t.fieldIdx("contig")
  private val startIdx: Int = t.fieldIdx("start")
  private val refIdx: Int = t.fieldIdx("ref")
  private val altAllelesIdx: Int = t.fieldIdx("altAlleles")
  private val tAltAlleles: TArray = t.fieldType(altAllelesIdx).asInstanceOf[TArray]
  private val tAltAllele: TStruct = tAltAlleles.elementType.asInstanceOf[TStruct]
  private val altIdx: Int = tAltAllele.fieldIdx("alt")

  assert(t.fieldType(contigIdx).required)
  assert(t.fieldType(startIdx).required)
  assert(t.fieldType(refIdx).required)
  assert(tAltAlleles.required)
  assert(tAltAlleles.elementType.required)
  assert(tAltAllele.fieldType(altIdx).required)

  def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  def setRegion(region: MemoryBuffer, offset: Long) {
    this.region = region
    this.offset = offset
    _contig = null
    _ref = null
    _altAlleles = null
  }

  private var _contig: String = null
  private var _ref: String = null
  private var _altAlleles: Array[AltAllele] = null

  def contig(): String = {
    if (_contig == null)
      _contig = TString.loadString(region, t.loadField(region, offset, contigIdx))
    _contig
  }

  def start(): Int = {
    region.loadInt(t.loadField(region, offset, startIdx))
  }

  def ref(): String = {
    if (_ref == null)
      _ref = TString.loadString(region, t.loadField(region, offset, refIdx))
    _ref
  }

  def altAlleles(): IndexedSeq[AltAllele] = {
    if (_altAlleles == null) {
      val aoff = t.loadField(region, offset, altAllelesIdx)
      val length = tAltAlleles.loadLength(region, aoff)
      val a = new Array[AltAllele](length)
      var i = 0
      while (i < length) {
        val eoff = tAltAlleles.loadElement(region, aoff, length, i)
        val alt = TString.loadString(region, tAltAllele.loadField(region, eoff, altIdx))
        val ref = TString.loadString(region, tAltAllele.loadField(region, eoff, refIdx))
        a(i) = AltAllele(alt, ref)
      }
      _altAlleles = a
    }
    _altAlleles
  }

  def copy(contig: String, start: Int, ref: String, altAlleles: IndexedSeq[AltAllele]): Variant =
    throw new UnsupportedOperationException("Copying a VariantView is expensive, don't do that!")
}
