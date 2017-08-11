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
  private val ta = rs.fields(2).typ.asInstanceOf[TArray]
  private val t = TGenotype.representation
  private val tArrayInt32 = TArray(TInt32)
  private val tSize = ta.elementByteSize

  private val gtIndex = 0
  private val gtOffset = t.byteOffsets(gtIndex)

  private val adIndex = 1
  private val adOffset = t.byteOffsets(adIndex)

  private val dpIndex = 2
  private val dpOffset = t.byteOffsets(dpIndex)

  private val gqIndex = 3
  private val gqOffset = t.byteOffsets(gqIndex)

  private val pxIndex = 4
  private val pxOffset = t.byteOffsets(pxIndex)

  private val linearScaleIndex = 6
  private val linearScaleOffset = t.byteOffsets(linearScaleIndex)

  private var m: MemoryBuffer = _
  private var gsOffset: Int = _
  private var gsLength: Int = _
  private var gOffset: Int = _
  private var gIsNull: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Int) {
    this.m = mb
    gsOffset = mb.loadInt(offset + rs.byteOffsets(2))
    gsLength = mb.loadInt(gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gOffset = ta.elementOffset(gsOffset, gsLength, idx)
    gIsNull = m.loadBit(gsOffset + 4, idx)
  }

  def hasGT: Boolean = !gIsNull && !m.loadBit(gOffset, gtIndex)

  def hasAD: Boolean = !gIsNull && !m.loadBit(gOffset, adIndex)

  def hasDP: Boolean = !gIsNull && !m.loadBit(gOffset, dpIndex)

  def hasGQ: Boolean = !gIsNull && !m.loadBit(gOffset, gqIndex)

  def hasPL: Boolean = !gIsNull && !m.loadBit(gOffset, pxIndex) && !m.loadBit(gOffset + linearScaleOffset, 0)

  def getGT: Int = Call(m.loadInt(gOffset + gtOffset))

  def getAD(idx: Int): Int = {
    val length = m.loadInt(gOffset + adOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(!m.loadBit(gOffset + adOffset + 4, idx))
    m.loadInt(tArrayInt32.elementOffset(gOffset + adOffset, length, idx))
  }

  def getDP: Int = m.loadInt(gOffset + dpOffset)

  def getGQ: Int = m.loadInt(gOffset + gqOffset)

  def getPL(idx: Int): Int = {
    val length = m.loadInt(gOffset + pxOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(!m.loadBit(gOffset + pxOffset + 4, idx))
    m.loadInt(tArrayInt32.elementOffset(gOffset + pxOffset, length, idx))
  }

}

object StructGenotypeView {
  val expectedGT = TCall
  val expectedAD = TArray(TInt32)
  val expectedDP = TInt32
  val expectedGQ = TInt32
  val expectedPL = TArray(TInt32)
}

private class StructGenotypeView(rs: TStruct) extends HTSGenotypeView {
  private val ta = rs.fields(2).typ.asInstanceOf[TArray]
  private val t = ta.elementType.asInstanceOf[TStruct]
  private val tArrayInt32 = TArray(TInt32)

  private def lookupField(name: String, expected: Type): (Boolean, Int, Int) = {
    t.fieldIdx.get(name) match {
      case Some(i) => (true, i, t.byteOffsets(i))
      case None => (false, uninitialized[Int], uninitialized[Int])
    }
  }

  private val (gtExists, gtIndex, gtOffset) = lookupField("GT", TCall)
  private val (adExists, adIndex, adOffset) = lookupField("AD", tArrayInt32)
  private val (dpExists, dpIndex, dpOffset) = lookupField("DP", TInt32)
  private val (gqExists, gqIndex, gqOffset) = lookupField("GQ", TInt32)
  private val (plExists, plIndex, plOffset) = lookupField("PL", tArrayInt32)

  private var m: MemoryBuffer = _
  private var gsOffset: Int = _
  private var gsLength: Int = _
  private var gOffset: Int = _
  private var gIsNull: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Int) {
    this.m = mb
    gsOffset = mb.loadInt(offset + rs.byteOffsets(2))
    gsLength = mb.loadInt(gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gOffset = ta.elementOffset(gsOffset, gsLength, idx)
    gIsNull = m.loadBit(gsOffset + 4, idx)
  }

  def hasGT: Boolean = gtExists && !gIsNull && !m.loadBit(gOffset, gtIndex)

  def hasAD: Boolean = adExists && !gIsNull && !m.loadBit(gOffset, adIndex)

  def hasDP: Boolean = dpExists && !gIsNull && !m.loadBit(gOffset, dpIndex)

  def hasGQ: Boolean = gqExists && !gIsNull && !m.loadBit(gOffset, gqIndex)

  def hasPL: Boolean = plExists && !gIsNull && !m.loadBit(gOffset, plIndex)

  def getGT: Int = Call(m.loadInt(gOffset + gtOffset))

  def getAD(idx: Int): Int = {
    val length = m.loadInt(gOffset + adOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(!m.loadBit(gOffset + adOffset + 4, idx))
    m.loadInt(tArrayInt32.elementOffset(gOffset + adOffset, length, idx))
  }

  def getDP: Int = m.loadInt(gOffset + dpOffset)

  def getGQ: Int = m.loadInt(gOffset + gqOffset)

  def getPL(idx: Int): Int = {
    val length = m.loadInt(gOffset + plOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(!m.loadBit(gOffset + plOffset + 4, idx))
    m.loadInt(tArrayInt32.elementOffset(gOffset + plOffset, length, idx))
  }
}
