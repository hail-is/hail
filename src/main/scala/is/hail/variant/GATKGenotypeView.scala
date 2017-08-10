package is.hail.variant

import is.hail.annotations.{MemoryBuffer, RegionValue, UnsafeRow, UnsafeUtils}
import is.hail.expr._
import is.hail.utils._

object GATKGenotypeView {
  def apply(rowSignature: TStruct): GATKGenotypeView = {
    rowSignature.fields(2).typ.asInstanceOf[TArray].elementType match {
      case TGenotype => new TGenotypeView(rowSignature)
      case _: TStruct => new StructGenotypeView(rowSignature)
      case t => fatal(s"invalid genotype representation: $t, expect TGenotype or TStruct")
    }
  }
}

// FIXME: This is removed, and StructGenotypeView becomes the only genotype view when TGenotype is removed
sealed abstract class GATKGenotypeView {
  def setRegion(mb: MemoryBuffer, offset: Int)

  def setGenotype(idx: Int)

  def hasGT: Boolean

  def hasAD: Boolean

  def hasDP: Boolean

  def hasGQ: Boolean

  def hasPL: Boolean

  def getGT: Int

  def getAD(idx: Int): Int

  def getDP: Int

  def getGQ: Int

  def getPL(idx: Int): Int
}

private class TGenotypeView(rs: TStruct) extends GATKGenotypeView {
  private val ta = rs.fields(2).typ.asInstanceOf[TArray]
  private val t = TGenotype.representation
  private val tSize = ta.elementByteSize
  private val gsOffset = rs.byteOffsets(2)

  private val gtIndex = t.fieldIdx("gt")
  private val gtOffset = t.byteOffsets(gtIndex)

  private val adIndex = t.fieldIdx("ad")
  private val adOffset = t.byteOffsets(adIndex)

  private val dpIndex = t.fieldIdx("dp")
  private val dpOffset = t.byteOffsets(dpIndex)

  private val gqIndex = t.fieldIdx("gq")
  private val gqOffset = t.byteOffsets(gqIndex)

  private val pxIndex = t.fieldIdx("px")
  private val pxOffset = t.byteOffsets(pxIndex)

  private val linearScaleIndex = t.fieldIdx("isLinearScale")
  private val linearScaleOffset = t.byteOffsets(linearScaleIndex)

  private var m: MemoryBuffer = _
  private var arrStart: Int = _
  private var arrSize: Int = _
  private var elementsStart: Int = _
  private var currentElementStart: Int = _
  private var elementIsNull: Boolean = _

  // -1 is used to indicate that the start and length must be computed
  private var plElemsStart = -1
  private var plArrayStart: Int = _
  private var plArrayLength: Int = _

  // -1 is used to indicate that the start and length must be computed
  private var adElemsStart = -1
  private var adArrayStart: Int = _
  private var adArrayLength: Int = _

  def setRegion(mb: MemoryBuffer, offset: Int) {
    this.m = mb
    arrStart = mb.loadInt(gsOffset + offset)
    arrSize = mb.loadInt(arrStart)
    elementsStart = arrStart + ta.elementsOffset(arrSize)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < arrSize)
    currentElementStart = elementsStart + idx * tSize
    elementIsNull = m.loadBit(arrStart + 4, idx)
    plElemsStart = -1
    adElemsStart = -1
  }

  def hasGT: Boolean = !elementIsNull && !m.loadBit(currentElementStart, gtIndex)

  def hasAD: Boolean = !elementIsNull && !m.loadBit(currentElementStart, adIndex)

  def hasDP: Boolean = !elementIsNull && !m.loadBit(currentElementStart, dpIndex)

  def hasGQ: Boolean = !elementIsNull && !m.loadBit(currentElementStart, gqIndex)

  def hasPL: Boolean = !elementIsNull && !m.loadBit(currentElementStart, pxIndex) && !m.loadBit(currentElementStart + linearScaleOffset, 0)

  def getGT: Int = Call(m.loadInt(currentElementStart + gtOffset))

  def getAD(idx: Int): Int = {
    if (adElemsStart == -1) {
      adArrayStart = m.loadInt(currentElementStart + adOffset)
      adArrayLength = m.loadInt(adArrayStart)
      adElemsStart = adArrayStart + StructGenotypeView.expectedAD.elementsOffset(adArrayStart)
    }
    if (idx < 0 || idx >= adArrayLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(!m.loadBit(adArrayStart + 4, idx))
    m.loadInt(adElemsStart + idx * 4)
  }

  def getDP: Int = m.loadInt(currentElementStart + dpOffset)

  def getGQ: Int = m.loadInt(currentElementStart + gqOffset)

  def getPL(idx: Int): Int = {
    if (plElemsStart == -1) {
      plArrayStart = m.loadInt(currentElementStart + pxOffset)
      plArrayLength = m.loadInt(plArrayStart)
      plElemsStart = plArrayStart + StructGenotypeView.expectedPL.elementsOffset(plArrayStart)
    }
    if (idx < 0 || idx >= plArrayLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(!m.loadBit(plArrayStart + 4, idx))
    m.loadInt(plElemsStart + idx * 4)
  }

}

object StructGenotypeView {
  val expectedGT = TCall
  val expectedAD = TArray(TInt32)
  val expectedDP = TInt32
  val expectedGQ = TInt32
  val expectedPL = TArray(TInt32)
}

private class StructGenotypeView(rs: TStruct) extends GATKGenotypeView {
  private val ta = rs.fields(2).typ.asInstanceOf[TArray]
  private val t = ta.elementType.asInstanceOf[TStruct]
  private val gsOffset = rs.byteOffsets(2)
  private val tSize = ta.elementByteSize

  private val hasFieldGT: Boolean = t.fieldIdx.contains("GT") && t.field("GT").typ == StructGenotypeView.expectedGT
  private val hasFieldAD: Boolean = t.fieldIdx.contains("AD") && t.field("AD").typ == StructGenotypeView.expectedAD
  private val hasFieldDP: Boolean = t.fieldIdx.contains("DP") && t.field("DP").typ == StructGenotypeView.expectedDP
  private val hasFieldGQ: Boolean = t.fieldIdx.contains("GQ") && t.field("GQ").typ == StructGenotypeView.expectedGQ
  private val hasFieldPL: Boolean = t.fieldIdx.contains("PL") && t.field("PL").typ == StructGenotypeView.expectedPL

  private val (gtIndex, gtOffset) = if (hasFieldGT) {
    val gtIndex = t.fieldIdx("GT")
    val gtOffset = t.byteOffsets(gtIndex)
    (gtIndex, gtOffset)
  } else (uninitialized[Int], uninitialized[Int])

  private val (adIndex, adOffset) = if (hasFieldAD) {
    val adIndex = t.fieldIdx("AD")
    val adOffset = t.byteOffsets(adIndex)
    (adIndex, adOffset)
  } else (uninitialized[Int], uninitialized[Int])

  private val (dpIndex, dpOffset) = if (hasFieldDP) {
    val dpIndex = t.fieldIdx("DP")
    val dpOffset = t.byteOffsets(dpIndex)
    (dpIndex, dpOffset)
  } else (uninitialized[Int], uninitialized[Int])

  private val (gqIndex, gqOffset) = if (hasFieldGQ) {
    val gqIndex = t.fieldIdx("GQ")
    val gqOffset = t.byteOffsets(gqIndex)
    (gqIndex, gqOffset)
  } else (uninitialized[Int], uninitialized[Int])

  private val (plIndex, plOffset) = if (hasFieldPL) {
    val plIndex = t.fieldIdx("PL")
    val plOffset = t.byteOffsets(plIndex)
    (plIndex, plOffset)
  } else (uninitialized[Int], uninitialized[Int])

  private var m: MemoryBuffer = _
  private var arrStart: Int = _
  private var arrSize: Int = _
  private var elementsStart: Int = _
  private var currentElementStart: Int = _
  private var elementIsNull: Boolean = _

  // -1 is used to indicate that the start and length must be computed
  private var plElemsStart = -1
  private var plArrayStart: Int = _
  private var plArrayLength: Int = _

  // -1 is used to indicate that the start and length must be computed
  private var adElemsStart = -1
  private var adArrayStart: Int = _
  private var adArrayLength: Int = _

  def setRegion(mb: MemoryBuffer, offset: Int) {
    this.m = mb
    arrStart = mb.loadInt(gsOffset + offset)
    arrSize = mb.loadInt(arrStart)
    elementsStart = arrStart + ta.elementsOffset(arrStart)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < arrSize)
    currentElementStart = elementsStart + idx * tSize
    elementIsNull = m.loadBit(arrStart + 4, idx)
    plElemsStart = -1
    adElemsStart = -1
  }

  def hasGT: Boolean = hasFieldGT && !elementIsNull && !m.loadBit(currentElementStart, gtIndex)

  def hasAD: Boolean = hasFieldAD && !elementIsNull && !m.loadBit(currentElementStart, adIndex)

  def hasDP: Boolean = hasFieldDP && !elementIsNull && !m.loadBit(currentElementStart, dpIndex)

  def hasGQ: Boolean = hasFieldGQ && !elementIsNull && !m.loadBit(currentElementStart, gqIndex)

  def hasPL: Boolean = hasFieldPL && !elementIsNull && !m.loadBit(currentElementStart, plIndex)

  def getGT: Int = Call(m.loadInt(currentElementStart + gtOffset))

  def getAD(idx: Int): Int = {
    if (adElemsStart == -1) {
      adArrayStart = m.loadInt(currentElementStart + adOffset)
      adArrayLength = m.loadInt(adArrayStart)
      adElemsStart = adArrayStart + StructGenotypeView.expectedAD.elementsOffset(adArrayLength)
    }
    if (idx < 0 || idx >= adArrayLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    // FIXME: When we have non-nullable types, ensure that these arrays are non-nullable and remove this check
    assert(!m.loadBit(adArrayStart + 4, idx))
    m.loadInt(adElemsStart + idx * 4)
  }

  def getDP: Int = m.loadInt(currentElementStart + dpOffset)

  def getGQ: Int = m.loadInt(currentElementStart + gqOffset)

  def getPL(idx: Int): Int = {
    if (plElemsStart == -1) {
      plArrayStart = m.loadInt(currentElementStart + plOffset)
      plArrayLength = m.loadInt(plArrayStart)
      plElemsStart = plArrayStart + StructGenotypeView.expectedPL.elementsOffset(plArrayLength)
    }
    if (idx < 0 || idx >= plArrayLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    // FIXME: When we have non-nullable types, ensure that these arrays are non-nullable and remove this check
    assert(!m.loadBit(plArrayStart + 4, idx))
    m.loadInt(plElemsStart + idx * 4)
  }
}
