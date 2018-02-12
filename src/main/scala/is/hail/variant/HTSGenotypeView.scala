package is.hail.variant

import java.util.zip.DataFormatException

import is.hail.annotations.{Region, RegionValue, UnsafeRow, UnsafeUtils}
import is.hail.expr.types._
import is.hail.utils._

object HTSGenotypeView {
  def apply(rowSignature: TStruct): HTSGenotypeView = {
    new HTSGenotypeView(rowSignature)
  }

  val tArrayInt32 = TArray(+TInt32())
}

final class HTSGenotypeView(rvType: TStruct) {
  private val entriesIndex = rvType.fieldByName(MatrixType.entriesIdentifier).index
  private val tgs = rvType.fieldType(entriesIndex).asInstanceOf[TArray]
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

  private var m: Region = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    gsOffset = rvType.loadField(m, offset, entriesIndex)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setRegion(rv: RegionValue) { setRegion(rv.region, rv.offset) }

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

  def getGT: Call = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    m.loadInt(callOffset)
  }

  def getADLength: Int = {
    val adOffset = tg.loadField(m, gOffset, adIndex)
    HTSGenotypeView.tArrayInt32.loadLength(m, adOffset)
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

  def getPLLength: Int = {
    val plOffset = tg.loadField(m, gOffset, plIndex)
    HTSGenotypeView.tArrayInt32.loadLength(m, plOffset)
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

final class ArrayGenotypeView(rvType: TStruct) {
  private val entriesIndex = rvType.fieldByName(MatrixType.entriesIdentifier).index
  private val tgs = rvType.fieldType(entriesIndex).asInstanceOf[TArray]
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

  private var m: Region = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    gsOffset = rvType.loadField(m, offset, entriesIndex)
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

  def getGT: Call = {
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    m.loadInt(callOffset)
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
    new HardCallView(rowSignature, "GT")
  }
}

final class HardCallView(rvType: TStruct, callField: String) {
  private val entriesIndex = rvType.fieldByName(MatrixType.entriesIdentifier).index
  private val tgs = rvType.fieldType(entriesIndex).asInstanceOf[TArray]
  private val tg = tgs.elementType.asInstanceOf[TStruct]

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

  private var m: Region = _
  private var gsOffset: Long = _
  private var gOffset: Long = _

  var gsLength: Int = _
  var gIsDefined: Boolean = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    gsOffset = rvType.loadField(m, offset, entriesIndex)
    gsLength = tgs.loadLength(m, gsOffset)
  }

  def setRegion(rv: RegionValue): Unit = setRegion(rv.region, rv.offset)

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(m, gsOffset, idx)
    gOffset = tgs.loadElement(m, gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gtExists && gIsDefined && tg.isFieldDefined(m, gOffset, gtIndex)

  def getGT: Call = {
    assert(gtExists && gIsDefined)
    val callOffset = tg.loadField(m, gOffset, gtIndex)
    m.loadInt(callOffset)
  }

  def getLength: Int = gsLength
}

final class HardcallTrioGenotypeView(rvType: TStruct, callField: String) {
  private val entriesIndex = rvType.fieldByName(MatrixType.entriesIdentifier).index
  private val trioTgs = rvType.fieldType(entriesIndex).asInstanceOf[TArray]
  private val trioTg = trioTgs.elementType.asInstanceOf[TStruct]

  require(trioTg.hasField("proband_entry") && trioTg.hasField("mother_entry") && trioTg.hasField("father_entry"))
  private val probandIndex = trioTg.fieldIdx("proband_entry")
  private val motherIndex = trioTg.fieldIdx("mother_entry")
  private val fatherIndex = trioTg.fieldIdx("father_entry")

  private val tg = trioTg.fieldType(probandIndex).fundamentalType.asInstanceOf[TStruct]

  require(tg.hasField(callField))
  private val gtIndex = tg.fieldIdx(callField)
  require(tg.unify(trioTg.fieldType(motherIndex).fundamentalType) && tg.unify(trioTg.fieldType(fatherIndex).fundamentalType))

  private var m: Region = _
  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _
  private var probandOffset: Long = _
  private var motherOffset: Long = _
  private var fatherOffset: Long = _

  var gIsDefined: Boolean = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    gsOffset = rvType.loadField(m, offset, entriesIndex)
    gsLength = trioTgs.loadLength(m, gsOffset)
  }

  def setRegion(rv: RegionValue) {
    setRegion(rv.region, rv.offset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = trioTgs.isElementDefined(m, gsOffset, idx)
    gOffset = trioTgs.loadElement(m, gsOffset, gsLength, idx)
    probandOffset = trioTg.loadField(m, gOffset, probandIndex)
    motherOffset = trioTg.loadField(m, gOffset, motherIndex)
    fatherOffset = trioTg.loadField(m, gOffset, fatherIndex)
  }

  def hasProbandGT: Boolean =
    gIsDefined &&
      trioTg.isFieldDefined(m, gOffset, probandIndex) &&
      tg.isFieldDefined(m, probandOffset, gtIndex)

  def hasMotherGT: Boolean =
    gIsDefined &&
      trioTg.isFieldDefined(m, gOffset, motherIndex) &&
      tg.isFieldDefined(m, motherOffset, gtIndex)

  def hasFatherGT: Boolean =
    gIsDefined &&
      trioTg.isFieldDefined(m, gOffset, fatherIndex) &&
      tg.isFieldDefined(m, fatherOffset, gtIndex)

  def hasAllGTs: Boolean =
    hasProbandGT && hasMotherGT && hasFatherGT

  def getProbandGT: Call = {
    val callOffset = tg.loadField(m, probandOffset, gtIndex)
    m.loadInt(callOffset)
  }

  def getMotherGT: Call = {
    val callOffset = tg.loadField(m, motherOffset, gtIndex)
    m.loadInt(callOffset)
  }

  def getFatherGT: Call = {
    val callOffset = tg.loadField(m, fatherOffset, gtIndex)
    m.loadInt(callOffset)
  }
}
