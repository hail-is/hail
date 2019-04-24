package is.hail.variant

import is.hail.annotations._
import is.hail.expr.types.physical.{PArray, PString, PStruct}
import is.hail.utils._

class RegionValueVariant(rowType: PStruct) extends View {
  private val locusField = rowType.fieldByName("locus")
  private val allelesField = rowType.fieldByName("alleles")
  private val locusIdx = locusField.index
  private val allelesIdx = allelesField.index
  private val tl: PStruct = locusField.typ.fundamentalType.asInstanceOf[PStruct]
  private val taa: PArray = allelesField.typ.asInstanceOf[PArray]
  private var region: Region = _
  private var locusOffset: Long = _
  private var allelesOffset: Long = _

  private var cachedContig: String = null
  private var cachedAlleles: Array[String] = null
  private var cachedLocus: Locus = null

  def setRegion(region: Region, offset: Long) {
    this.region = region

    if (!rowType.isFieldDefined(region, offset, locusIdx))
      fatal(s"The row field 'locus' cannot have missing values.")
    if (!rowType.isFieldDefined(region, offset, allelesIdx))
      fatal(s"The row field 'alleles' cannot have missing values.")
    this.locusOffset = rowType.loadField(region, offset, locusIdx)
    this.allelesOffset = rowType.loadField(region, offset, allelesIdx)
    cachedContig = null
    cachedAlleles = null
    cachedLocus = null
  }

  def contig(): String = {
    if (cachedContig == null)
      cachedContig = PString.loadString(region, tl.loadField(region, locusOffset, 0))
    cachedContig
  }

  def position(): Int = {
    region.loadInt(tl.loadField(region, locusOffset, 1))
  }

  def alleles(): Array[String] = {
    if (cachedAlleles == null) {
      val nAlleles = taa.loadLength(region, allelesOffset)
      cachedAlleles = new Array[String](nAlleles)
      var i = 0
      while (i < nAlleles) {
        if (taa.isElementDefined(region, allelesOffset, i))
         cachedAlleles(i) = PString.loadString(region, taa.loadElement(region, allelesOffset, i))
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
