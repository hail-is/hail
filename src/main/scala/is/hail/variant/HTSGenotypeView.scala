package is.hail.variant

import is.hail.annotations.{MemoryBuffer, RegionValue, UnsafeRow, UnsafeUtils}
import is.hail.expr._
import is.hail.utils._

object HTSGenotypeView {
  def apply(rowSignature: TStruct): HTSGenotypeView = {
    rowSignature.fields(2).typ.asInstanceOf[TArray].elementType match {
      case TGenotype => new TGenotypeView(rowSignature)
      case _: TStruct => new StructGenotypeView(rowSignature)
      case t => fatal(s"invalid genotype representation: $t, expect TGenotype or TStruct")
    }
  }

  val tArrayInt32 = TArray(TInt32)
}

// FIXME: This is removed, and StructGenotypeView becomes the only genotype view when TGenotype is removed
sealed abstract class HTSGenotypeView {
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

private class TGenotypeView(rs: TStruct) extends HTSGenotypeView {
  private val tgs = rs.fields(2).typ.asInstanceOf[TArray]
  private val tg = TGenotype.representation

  private val gtIndex = 0
  private val adIndex = 1
  private val dpIndex = 2
  private val gqIndex = 3
  private val pxIndex = 4
  private val linearScaleIndex = 6

  private var m: MemoryBuffer = _
  private var gsOffset: Int = _
  private var gsLength: Int = _
  private var gOffset: Int = _
  private var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Int) {
    this.m = mb
    gsOffset = rs.loadField(m, offset, 2)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(m, gsOffset, idx)
    gOffset = tgs.loadElement(m, gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, gtIndex)

  def hasAD: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, adIndex)

  def hasDP: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, dpIndex)

  def hasGQ: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, gqIndex)

  def hasPL: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, pxIndex) &&
    m.loadBoolean(tg.loadField(m, gOffset, linearScaleIndex))

  def getGT: Int = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    Call(m.loadInt(callOffset))
  }

  def getAD(idx: Int): Int = {
    val adOffset = tg.loadField(m, gOffset, adIndex)
    val length = HTSGenotypeView.tArrayInt32.loadLength(m, adOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(HTSGenotypeView.tArrayInt32.isElementDefined(m, adOffset, idx))

    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(gOffset + adOffset, length, idx)
    m.loadInt(elementOffset)
  }

  def getDP: Int = {
    val dpOffset = tg.loadField(m, gOffset, dpIndex)
    m.loadInt(dpOffset)
  }

  def getGQ: Int = {
    val gqOffset = tg.loadField(m, gOffset, gqIndex)
    m.loadInt(gqOffset)
  }

  def getPL(idx: Int): Int = {
    val pxOffset = tg.loadField(m, gOffset, pxIndex)
    val length = HTSGenotypeView.tArrayInt32.loadLength(m, pxOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(HTSGenotypeView.tArrayInt32.isElementDefined(m, pxOffset, idx))
    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(gOffset + pxOffset, length, idx)
    m.loadInt(elementOffset)
  }
}

private class StructGenotypeView(rs: TStruct) extends HTSGenotypeView {
  private val tgs = rs.fields(2).asInstanceOf[TArray]
  private val tg = tgs.elementType.asInstanceOf[TStruct]

  private def lookupField(name: String, expected: Type): (Boolean, Int) = {
    tg.fieldIdx.get(name) match {
      case Some(i) => (true, i)
      case None => (false, uninitialized[Int])
    }
  }

  private val (gtExists, gtIndex) = lookupField("GT", TCall)
  private val (adExists, adIndex) = lookupField("AD", HTSGenotypeView.tArrayInt32)
  private val (dpExists, dpIndex) = lookupField("DP", TInt32)
  private val (gqExists, gqIndex) = lookupField("GQ", TInt32)
  private val (plExists, plIndex) = lookupField("PL", HTSGenotypeView.tArrayInt32)

  private var m: MemoryBuffer = _
  private var gsOffset: Int = _
  private var gsLength: Int = _
  private var gOffset: Int = _
  private var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Int) {
    this.m = mb
    gsOffset = rs.loadField(m, offset, 2)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(m, gsOffset, idx)
    gOffset = tgs.loadElement(m, gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gtExists && gIsDefined && tg.isFieldDefined(m, gOffset, gtIndex)

  def hasAD: Boolean = adExists && gIsDefined && tg.isFieldDefined(m, gOffset, adIndex)

  def hasDP: Boolean = dpExists && gIsDefined && tg.isFieldDefined(m, gOffset, dpIndex)

  def hasGQ: Boolean = gqExists && gIsDefined && tg.isFieldDefined(m, gOffset, gqIndex)

  def hasPL: Boolean = plExists && gIsDefined && tg.isFieldDefined(m, gOffset, plIndex)

  def getGT: Int = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    Call(m.loadInt(callOffset))
  }

  def getAD(idx: Int): Int = {
    val adOffset = tg.loadField(m, gOffset, adIndex)
    val length = HTSGenotypeView.tArrayInt32.loadLength(m, adOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(HTSGenotypeView.tArrayInt32.isElementDefined(m, adOffset, idx))

    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(gOffset + adOffset, length, idx)
    m.loadInt(elementOffset)
  }

  def getDP: Int = {
    val dpOffset = tg.loadField(m, gOffset, dpIndex)
    m.loadInt(dpOffset)
  }

  def getGQ: Int = {
    val gqOffset = tg.loadField(m, gOffset, gqIndex)
    m.loadInt(gqOffset)
  }

  def getPL(idx: Int): Int = {
    val pxOffset = tg.loadField(m, gOffset, plIndex)
    val length = HTSGenotypeView.tArrayInt32.loadLength(m, pxOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(HTSGenotypeView.tArrayInt32.isElementDefined(m, pxOffset, idx))
    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(gOffset + pxOffset, length, idx)
    m.loadInt(elementOffset)
  }
}
