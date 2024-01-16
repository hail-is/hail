package is.hail.variant
import is.hail.types.physical.{PArray, PInt32, PLocus, PString, PStruct}
import is.hail.utils._

class RegionValueVariant(rowType: PStruct) extends View {
  private val locusField = rowType.fieldByName("locus")
  private val locusPType = locusField.typ
  private val allelesField = rowType.fieldByName("alleles")
  private val locusIdx = locusField.index
  private val allelesIdx = allelesField.index
  private val taa: PArray = allelesField.typ.asInstanceOf[PArray]
  private val allelePType = taa.elementType.asInstanceOf[PString]
  private var locusAddress: Long = _
  private var allelesOffset: Long = _
  private var cachedContig: String = null
  private var cachedAlleles: Array[String] = null
  private var cachedLocus: Locus = null

  def set(address: Long): Unit = {
    if (!rowType.isFieldDefined(address, locusIdx))
      fatal(s"The row field 'locus' cannot have missing values.")
    if (!rowType.isFieldDefined(address, allelesIdx))
      fatal(s"The row field 'alleles' cannot have missing values.")
    this.locusAddress = rowType.loadField(address, locusIdx)
    this.allelesOffset = rowType.loadField(address, allelesIdx)
    cachedContig = null
    cachedAlleles = null
    cachedLocus = null
  }

  def contig(): String = {
    if (cachedContig == null) {
      locusPType match {
        case pl: PLocus =>
          cachedContig = pl.contig(locusAddress)
        case s: PStruct =>
          cachedContig = s.types(0).asInstanceOf[PString].loadString(s.loadField(locusAddress, 0))
      }
    }
    cachedContig
  }

  def position(): Int = locusPType match {
    case pl: PLocus =>
      pl.position(locusAddress)
    case s: PStruct =>
      s.types(1).asInstanceOf[PInt32].unstagedLoadFromAddress(s.loadField(locusAddress, 1))
  }

  def alleles(): Array[String] = {
    if (cachedAlleles == null) {
      val nAlleles = taa.loadLength(allelesOffset)
      cachedAlleles = new Array[String](nAlleles)
      var i = 0
      while (i < nAlleles) {
        if (taa.isElementDefined(allelesOffset, i))
          cachedAlleles(i) = allelePType.loadString(taa.loadElement(allelesOffset, i))
        i += 1
      }
    }
    cachedAlleles
  }

  def locus(): Locus = {
    if (cachedLocus == null) {
      cachedLocus = new Locus(contig(), position())
    }
    cachedLocus
  }
}
