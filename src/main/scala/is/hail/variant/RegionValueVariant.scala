package is.hail.variant

import is.hail.utils._

import is.hail.annotations._
import is.hail.expr._

class RegionValueVariant(tv: TVariant) extends Variant with View {
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
    cachedContig = null
    cachedRef = null
    cachedAltAlleles = null
    regionValueAltAlleles.setRegion(region, t.loadField(region, offset, altAllelesIdx))
  }

  def getOffset(): Long = offset

  private var cachedContig: String = null
  private var cachedRef: String = null
  private var cachedAltAlleles: Array[ConcreteAltAllele] = null

  override def contig(): String = {
    if (cachedContig == null)
      cachedContig = TString.loadString(region, t.loadField(region, offset, contigIdx))
    cachedContig
  }

  override def start(): Int = {
    region.loadInt(t.loadField(region, offset, startIdx))
  }

  override def ref(): String = {
    if (cachedRef == null)
      cachedRef = TString.loadString(region, t.loadField(region, offset, refIdx))
    cachedRef
  }

  def copy(contig: String, start: Int, ref: String, altAlleles: VolatileIndexedSeq[_ <: AltAllele]): Variant = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(tv.fundamentalType)
    rvb.startStruct()
    rvb.addString(contig)
    rvb.addInt(start)
    rvb.addString(ref)
    rvb.startArray(altAlleles.length)
    var i = 0
    while (i < altAlleles.length) {
      rvb.startStruct()
      val aa = altAlleles(i)
      rvb.addString(aa.ref)
      rvb.addString(aa.alt)
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    val rvr = new RegionValueVariant(tv)
    rvr.setRegion(region, rvb.end())
    rvr
  }

  def copy2(contig: String, start: Int, ref: String, altAlleles: VolatileIndexedSeq[String]): Variant = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(tv.fundamentalType)
    rvb.startStruct()
    rvb.addString(contig)
    rvb.addInt(start)
    rvb.addString(ref)
    rvb.startArray(altAlleles.length)
    var i = 0
    while (i < altAlleles.length) {
      rvb.startStruct()
      val aa = altAlleles(i)
      rvb.addString(ref)
      rvb.addString(aa)
      rvb.endStruct()
      i += 1
    }
    rvb.endArray()
    rvb.endStruct()
    val rvr = new RegionValueVariant(tv)
    rvr.setRegion(region, rvb.end())
    rvr
  }

  val regionValueAltAlleles = new ArrayView(tAltAlleles, new RegionValueAltAllele(tAltAllele))

  override def altAlleles(): IndexedSeq[ConcreteAltAllele] = {
    if (cachedAltAlleles == null)
      cachedAltAlleles = regionValueAltAlleles.toArray[ConcreteAltAllele](_.reify)
    cachedAltAlleles
  }
}
