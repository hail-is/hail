package is.hail.variant

import is.hail.annotations.{MemoryBuffer, RegionValue, UnsafeRow, UnsafeUtils}
import is.hail.expr._

final class GATKGenotypeView(@transient vsm: VariantSampleMatrix[_, _, _]) extends Serializable {
  // FIXME: When TGenotype is removed, this should always be a struct and fundamentalType will be unnecessary
  @transient private val t = vsm.genotypeSignature.fundamentalType.asInstanceOf[TStruct]
  private val tAlignment = t.alignment
  private val tSize = t.byteSize

  // FIXME: When TGenotype is removed, this should be TCall
  val hasAnyGT: Boolean = t.fieldIdx.contains("gt") && t.field("gt").typ == TInt32
  val hasAnyAD: Boolean = t.fieldIdx.contains("ad") && t.field("ad").typ == TArray(TInt32)
  val hasAnyDP: Boolean = t.fieldIdx.contains("dp") && t.field("dp").typ == TInt32
  val hasAnyGQ: Boolean = t.fieldIdx.contains("gq") && t.field("gq").typ == TInt32
  val hasAnyPL: Boolean = t.fieldIdx.contains("pl") && t.field("pl").typ == TArray(TInt32)

  private val gsOffset = vsm.rowSignature.byteOffsets(vsm.rowSignature.fieldIdx("gs"))
  private val tArrayOffset = UnsafeUtils.arrayElementSize(t)
  private val nSamples = vsm.nSamples
  private var iGT: Int = _
  private var iAD: Int = _
  private var iDP: Int = _
  private var iGQ: Int = _
  private var iPL: Int = _
  private var oGT: Int = _
  private var oAD: Int = _
  private var oDP: Int = _
  private var oGQ: Int = _
  private var oPL: Int = _

  if (hasAnyGT) {
    iGT = t.fieldIdx("gt")
    oGT = t.byteOffsets(iGT)
  }
  if (hasAnyAD) {
    iAD = t.fieldIdx("ad")
    oAD = t.byteOffsets(iAD)
  }
  if (hasAnyDP) {
    iDP = t.fieldIdx("dp")
    oDP = t.byteOffsets(iDP)
  }
  if (hasAnyGQ) {
    iGQ = t.fieldIdx("gq")
    oGQ = t.byteOffsets(iGQ)
  }
  if (hasAnyPL) {
    iPL = t.fieldIdx("pl")
    oPL = t.byteOffsets(iPL)
  }

  private var mb: MemoryBuffer = _
  private var arrStart: Int = _
  private var eltsStart: Int = _
  private var thisStart: Int = _
  private var eltNull: Boolean = _
  private var plElemsStart = -1
  private var plArrayLength: Int = _
  private var adElemsStart = -1
  private var adArrayLength: Int = _

  def setRegion(mb: MemoryBuffer) {
    this.mb = mb
    arrStart = mb.loadInt(gsOffset)
    eltsStart = UnsafeUtils.roundUpAlignment(arrStart + 4 + (nSamples + 7) / 8, tAlignment)

    assert(mb.loadInt(arrStart) == nSamples)
  }

  def set(idx: Int) {
    require(idx >= 0 && idx < nSamples)
    thisStart = eltsStart + idx * tSize
    eltNull = mb.loadBit(arrStart + 4, idx)
    plElemsStart = -1
    adElemsStart = -1
  }

  def hasGT: Boolean = hasAnyGT && !eltNull && !mb.loadBit(thisStart, iGT)

  def hasAD: Boolean = hasAnyAD && !eltNull && !mb.loadBit(thisStart, iAD)

  def hasDP: Boolean = hasAnyDP && !eltNull && !mb.loadBit(thisStart, iDP)

  def hasGQ: Boolean = hasAnyGQ && !eltNull && !mb.loadBit(thisStart, iGQ)

  def hasPL: Boolean = hasAnyPL && !eltNull && !mb.loadBit(thisStart, iPL)

  def getGT: Int = mb.loadInt(thisStart + oGT)

  def getAD(idx: Int): Int = {
    if (adElemsStart == -1) {
      val adArrStart = mb.loadInt(thisStart + oAD)
      adArrayLength = mb.loadInt(adElemsStart)
      adElemsStart = UnsafeUtils.roundUpAlignment(adArrStart + 4 + (adArrayLength + 7) / 8, 4)
    }
    if (idx < 0 || idx >= adArrayLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    mb.loadInt(adElemsStart + idx * 4)
  }

  def getDP: Int = mb.loadInt(thisStart + oDP)

  def getGQ: Int = mb.loadInt(thisStart + oGQ)

  def getPL(idx: Int): Int = {
    if (plElemsStart == -1) {
      val plArrStart = mb.loadInt(thisStart + oPL)
      plArrayLength = mb.loadInt(plElemsStart)
      plElemsStart = UnsafeUtils.roundUpAlignment(plArrStart + 4 + (plArrayLength + 7) / 8, 4)
    }
    if (idx < 0 || idx >= plArrayLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    mb.loadInt(plElemsStart + idx * 4)
  }
}
