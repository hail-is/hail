package is.hail.variant

import is.hail.annotations._
import is.hail.expr._

class VariantView(tv: TVariant) extends Variant with View {
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
    _contig = null
    _ref = null
    _altAlleles = null
    altAlleles.setRegion(region, t.loadField(region, offset, altAllelesIdx))
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

  def copy(contig: String, start: Int, ref: String, altAlleles: IndexedSeq[AltAllele]): Variant =
    throw new UnsupportedOperationException("Copying a VariantView is expensive, don't do that!")

  val altAlleles = new ArrayView(tAltAlleles, new AltAlleleView(tAltAllele))
}
