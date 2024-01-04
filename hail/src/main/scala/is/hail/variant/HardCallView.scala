package is.hail.variant

import is.hail.annotations.Region
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual.TCall

final class ArrayGenotypeView(rvType: PStruct) {
  private val entriesIndex = rvType.fieldByName(MatrixType.entriesIdentifier).index
  private val tgs = rvType.types(entriesIndex).asInstanceOf[PArray]
  private val tg = tgs.elementType.asInstanceOf[PStruct]

  private def lookupField(name: String, pred: PType => Boolean): (Boolean, Int, PType) = {
    tg.selfField(name) match {
      case Some(f) =>
        if (pred(f.typ))
          (true, f.index, f.typ)
        else
          (false, 0, null)
      case None =>
        (false, 0, null)
    }
  }

  private val (gtExists, gtIndex, _) = lookupField("GT", _ == PCanonicalCall())

  private val (gpExists, gpIndex, _gpType) = lookupField(
    "GP",
    pt => pt.isInstanceOf[PArray] && pt.asInstanceOf[PArray].elementType.isInstanceOf[PFloat64],
  )

  // Do not try to move this cast into the destructuring above
  /* https://stackoverflow.com/questions/27789412/scala-exception-in-for-comprehension-with-type-annotation */
  private[this] val gpType = _gpType.asInstanceOf[PArray]

  private var gsOffset: Long = _
  private var gsLength: Int = _
  private var gOffset: Long = _
  var gIsDefined: Boolean = _

  def set(offset: Long) {
    gsOffset = rvType.loadField(offset, entriesIndex)
    gsLength = tgs.loadLength(gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(gsOffset, idx)
    gOffset = tgs.loadElement(gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gtExists && gIsDefined && tg.isFieldDefined(gOffset, gtIndex)

  def hasGP: Boolean = gpExists && gIsDefined && tg.isFieldDefined(gOffset, gpIndex)

  def getGT: Call = {
    val callOffset = tg.loadField(gOffset, gtIndex)
    Region.loadInt(callOffset)
  }

  def getGP(idx: Int): Double = {
    val gpOffset = tg.loadField(gOffset, gpIndex)
    val length = gpType.loadLength(gpOffset)
    if (idx < 0 || idx >= length)
      throw new ArrayIndexOutOfBoundsException(idx)
    assert(gpType.isElementDefined(gpOffset, idx))
    val elementOffset = gpType.elementOffset(gpOffset, length, idx)
    Region.loadDouble(elementOffset)
  }

  def getGPLength(): Int = {
    val gpOffset = tg.loadField(gOffset, gpIndex)
    gpType.loadLength(gpOffset)
  }
}

object HardCallView {
  def apply(rowSignature: PStruct): HardCallView =
    new HardCallView(rowSignature, "GT")
}

final class HardCallView(rvType: PStruct, callField: String) {
  private val entriesIndex = rvType.fieldByName(MatrixType.entriesIdentifier).index
  private val tgs = rvType.types(entriesIndex).asInstanceOf[PArray]
  private val tg = tgs.elementType.asInstanceOf[PStruct]

  private val (gtExists, gtIndex) = {
    tg.selfField(callField) match {
      case Some(f) =>
        if (f.typ.virtualType == TCall)
          (true, f.index)
        else
          (false, 0)
      case None => (false, 0)
    }
  }

  private var gsOffset: Long = _
  private var gOffset: Long = _

  var gsLength: Int = _
  var gIsDefined: Boolean = _

  def set(offset: Long) {
    gsOffset = rvType.loadField(offset, entriesIndex)
    gsLength = tgs.loadLength(gsOffset)
  }

  def setGenotype(idx: Int) {
    require(idx >= 0 && idx < gsLength)
    gIsDefined = tgs.isElementDefined(gsOffset, idx)
    gOffset = tgs.loadElement(gsOffset, gsLength, idx)
  }

  def hasGT: Boolean = gtExists && gIsDefined && tg.isFieldDefined(gOffset, gtIndex)

  def getGT: Call = {
    assert(gtExists && gIsDefined)
    val callOffset = tg.loadField(gOffset, gtIndex)
    Region.loadInt(callOffset)
  }

  def getLength: Int = gsLength
}
