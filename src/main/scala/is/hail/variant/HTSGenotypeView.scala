package is.hail.variant

import java.util.zip.DataFormatException

import is.hail.annotations.{MemoryBuffer, RegionValue, UnsafeRow, UnsafeUtils}
import is.hail.expr._
import is.hail.utils._

object HTSGenotypeView {
  def apply(rowSignature: TStruct): HTSGenotypeView = {
    rowSignature.fieldType(3).asInstanceOf[TArray].elementType match {
      case _: TGenotype => new TGenotypeView(rowSignature)
      case _: TStruct => new StructGenotypeView(rowSignature)
      case t => fatal(s"invalid genotype representation: $t, expect TGenotype or TStruct")
    }
  }

  val tArrayInt32 = TArray(!TInt32())
}

// FIXME: This is removed, and StructGenotypeView becomes the only genotype view when TGenotype is removed
sealed abstract class HTSGenotypeView {
  def setRegion(mb: MemoryBuffer, offset: Long)

  def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  def setGenotype(idx: Int)

  def gIsDefined: Boolean

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

class TGenotypeView(rs: TStruct) extends HTSGenotypeView {
  private val tgs = rs.fieldType(3).asInstanceOf[TArray]
  private val tg = TGenotype.representation()

  private val gtIndex = 0
  private val adIndex = 1
  private val dpIndex = 2
  private val gqIndex = 3
  private val plIndex = 4

  private var m: MemoryBuffer = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    gsOffset = rs.loadField(m, offset, 3)
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

  def hasPL: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, plIndex)

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

    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(adOffset, length, idx)
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
    val plOffset = tg.loadField(m, gOffset, plIndex)
    val length = HTSGenotypeView.tArrayInt32.loadLength(m, plOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(HTSGenotypeView.tArrayInt32.isElementDefined(m, plOffset, idx))
    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(plOffset, length, idx)
    m.loadInt(elementOffset)
  }
}

private class StructGenotypeView(rs: TStruct) extends HTSGenotypeView {
  private val tgs = rs.fieldType(3).asInstanceOf[TArray]
  private val tg = tgs.elementType.asInstanceOf[TStruct]

  private def lookupField(name: String, expected: Type): (Boolean, Int) = {
    tg.selfField(name) match {
      case Some(f) =>
        if (f.typ == expected)
          (true, f.index)
        else
          (false, 0)
      case None => (false, 0)
    }
  }

  private val (gtExists, gtIndex) = lookupField("GT", TCall())
  private val (adExists, adIndex) = lookupField("AD", HTSGenotypeView.tArrayInt32)
  private val (dpExists, dpIndex) = lookupField("DP", TInt32())
  private val (gqExists, gqIndex) = lookupField("GQ", TInt32())
  private val (plExists, plIndex) = lookupField("PL", HTSGenotypeView.tArrayInt32)

  private var m: MemoryBuffer = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    gsOffset = rs.loadField(m, offset, 3)
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

    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(adOffset, length, idx)
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
    val plOffset = tg.loadField(m, gOffset, plIndex)
    val length = HTSGenotypeView.tArrayInt32.loadLength(m, plOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(HTSGenotypeView.tArrayInt32.isElementDefined(m, plOffset, idx))
    val elementOffset = HTSGenotypeView.tArrayInt32.elementOffset(plOffset, length, idx)
    m.loadInt(elementOffset)
  }
}

object ArrayGenotypeView {
  val tArrayFloat64 = TArray(TFloat64())
}

class ArrayGenotypeView(rowType: TStruct) {
  private val tgs = rowType.fieldType(3).asInstanceOf[TArray]
  private val tg = tgs.elementType match {
    case tg: TStruct => tg
    case _ => null
  }

  private def lookupField(name: String, expected: Type): (Boolean, Int) = {
    if (tg != null) {
      tg.selfField(name) match {
        case Some(f) =>
          if (f.typ == expected)
            (true, f.index)
          else
            (false, 0)
        case None => (false, 0)
      }
    } else
      (false, 0)
  }

  private val (gtExists, gtIndex) = lookupField("GT", TCall())
  private val (gpExists, gpIndex) = lookupField("GP", ArrayGenotypeView.tArrayFloat64)

  private var m: MemoryBuffer = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    gsOffset = rowType.loadField(m, offset, 3)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setRegion(rv: RegionValue): Unit = setRegion(rv.region, rv.offset)

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(m, gsOffset, idx)
    gOffset = tgs.loadElement(m, gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gtExists && gIsDefined && tg.isFieldDefined(m, gOffset, gtIndex)

  def hasGP: Boolean = gpExists && gIsDefined && tg.isFieldDefined(m, gOffset, gpIndex)

  def getGT: Int = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    Call(m.loadInt(callOffset))
  }

  def getGP(idx: Int): Double = {
    val adOffset = tg.loadField(m, gOffset, gpIndex)
    val length = ArrayGenotypeView.tArrayFloat64.loadLength(m, adOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(ArrayGenotypeView.tArrayFloat64.isElementDefined(m, adOffset, idx))

    val elementOffset = ArrayGenotypeView.tArrayFloat64.elementOffset(adOffset, length, idx)
    m.loadDouble(elementOffset)
  }
}

object HardCallView {
  def apply(rowSignature: TStruct): HardCallView = {
    rowSignature.fieldType(3).asInstanceOf[TArray].elementType match {
      case _: TGenotype => new HardCallTGenotypeView(rowSignature)
      case _: TStruct => new HardCallStructView(rowSignature, "GT")
      case t => fatal(s"invalid genotype representation: $t, expect TGenotype or TStruct")
    }
  }
}

abstract class HardCallView {
  def setRegion(mb: MemoryBuffer, offset: Long)

  def setRegion(rv: RegionValue): Unit = setRegion(rv.region, rv.offset)

  def setGenotype(idx: Int)

  def hasGT: Boolean

  def getGT: Int
}

class HardCallTGenotypeView(rowType: TStruct) extends HardCallView {
  private val tgs = rowType.fieldType(3).asInstanceOf[TArray]
  private val tg = TGenotype.representation()

  private val gtIndex = 0

  private var m: MemoryBuffer = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    gsOffset = rowType.loadField(m, offset, 3)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(m, gsOffset, idx)
    gOffset = tgs.loadElement(m, gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gIsDefined && tg.isFieldDefined(m, gOffset, gtIndex)

  def getGT: Int = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    val gt = m.loadInt(callOffset)
    if (gt < 0)
      throw new DataFormatException(s"Expected call to be non-negative, but found $gt")
    gt
  }
}

class HardCallStructView(rowType: TStruct, callField: String) extends HardCallView {
  private val tgs = rowType.fieldType(3).asInstanceOf[TArray]
  private val tg = tgs.elementType match {
    case tg: TStruct => tg
    case _ => null
  }

  private def lookupField(name: String, expected: Type): (Boolean, Int) = {
    if (tg != null) {
      tg.selfField(name) match {
        case Some(f) =>
          if (f.typ == expected)
            (true, f.index)
          else
            (false, 0)
        case None => (false, 0)
      }
    } else
      (false, 0)
  }

  private val (gtExists, gtIndex) = lookupField(callField, TCall())

  private var m: MemoryBuffer = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    gsOffset = rowType.loadField(m, offset, 3)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(m, gsOffset, idx)
    gOffset = tgs.loadElement(m, gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gtExists && gIsDefined && tg.isFieldDefined(m, gOffset, gtIndex)

  def getGT: Int = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    val gt = m.loadInt(callOffset)
    if (gt < 0)
      throw new DataFormatException(s"Expected call to be non-negative, but found $gt")
    gt
  }
}

class HardcallTrioGenotypeView(rs: TStruct, callField: String) {
  private val trioTgs = rs.fieldType(3).asInstanceOf[TArray]
  private val trioTg = trioTgs.elementType.asInstanceOf[TStruct]

  require(trioTg.hasField("proband") && trioTg.hasField("mother") && trioTg.hasField("father"))
  private val probandIndex = trioTg.fieldIdx("proband")
  private val motherIndex = trioTg.fieldIdx("mother")
  private val fatherIndex = trioTg.fieldIdx("father")

  private val tg = trioTg.fieldType(probandIndex).fundamentalType.asInstanceOf[TStruct]

  require(tg.hasField(callField))
  private val gtIndex = tg.fieldIdx(callField)
  require(tg.unify(trioTg.fieldType(motherIndex).fundamentalType) && tg.unify(trioTg.fieldType(fatherIndex).fundamentalType))

  private var m: MemoryBuffer = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _
  private var gProbandOffset: Long = _
  private var gMotherOffset: Long = _
  private var gFatherOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    gsOffset = rs.loadField(m, offset, 3)
    gsLength = trioTgs.loadLength(m, gsOffset)
  }

  def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = trioTgs.isElementDefined(m, gsOffset, idx)
    gOffset = trioTgs.loadElement(m, gsOffset, gsLength, idx)
    gProbandOffset = trioTg.loadField(m, gOffset, probandIndex)
    gMotherOffset = trioTg.loadField(m, gOffset, motherIndex)
    gFatherOffset = trioTg.loadField(m, gOffset, fatherIndex)
  }

  def hasProbandGT: Boolean =
    gIsDefined &&
      trioTg.isFieldDefined(m, gOffset, probandIndex) &&
      tg.isFieldDefined(m, gProbandOffset, gtIndex)

  def hasMotherGT: Boolean =
    gIsDefined &&
      trioTg.isFieldDefined(m, gOffset, motherIndex) &&
      tg.isFieldDefined(m, gMotherOffset, gtIndex)

  def hasFatherGT: Boolean =
    gIsDefined &&
      trioTg.isFieldDefined(m, gOffset, fatherIndex) &&
      tg.isFieldDefined(m, gFatherOffset, gtIndex)

  def hasAllGTs: Boolean =
    hasProbandGT && hasMotherGT && hasFatherGT

  def getProbandGT: Int = {
    val callOffset = tg.loadField(m, gProbandOffset, gtIndex)
    val gt = m.loadInt(callOffset)
    if (gt < 0)
      throw new DataFormatException(s"Expected call to be non-negative, but found $gt")
    gt
  }

  def getMotherGT: Int = {
    val callOffset = tg.loadField(m, gMotherOffset, gtIndex)
    val gt = m.loadInt(callOffset)
    if (gt < 0)
      throw new DataFormatException(s"Expected call to be non-negative, but found $gt")
    gt
  }

  def getFatherGT: Int = {
    val callOffset = tg.loadField(m, gFatherOffset, gtIndex)
    val gt = m.loadInt(callOffset)
    if (gt < 0)
      throw new DataFormatException(s"Expected call to be non-negative, but found $gt")
    gt
  }
}
