package is.hail.variant

import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class RegionValueAltAllele(taa: TAltAllele) extends View {
  private val t = taa.fundamentalType.asInstanceOf[TStruct]
  private val refIdx = t.fieldIdx("ref")
  private val altIdx = t.fieldIdx("alt")
  private var region: MemoryBuffer = _
  private var offset: Long = _
  private var _ref: String = null
  private var _alt: String = null

  assert(t.isFieldRequired(refIdx))
  assert(t.isFieldRequired(altIdx))

  def setRegion(region: MemoryBuffer, offset: Long) {
    this.region = region
    this.offset = offset
    this._ref = null
    this._alt = null
  }

  def getRef(): String = {
    if (_ref == null)
      _ref = TString.loadString(region, t.loadField(region, offset, refIdx))
    _ref
  }

  def getAlt(): String = {
    if (_alt == null)
      _alt = TString.loadString(region, t.loadField(region, offset, altIdx))
    _alt
  }

  import AltAlleleType._

  def altAlleleType: AltAlleleType = {
    if (isSNP)
      SNP
    else if (isInsertion)
      Insertion
    else if (isDeletion)
      Deletion
    else if (isStar)
      Star
    else if (getRef().length == getAlt().length)
      MNP
    else
      Complex
  }

  def isStar: Boolean = getAlt() == "*"

  def isSNP: Boolean = !isStar && ((getRef().length == 1 && getAlt().length == 1) ||
    (getRef().length == getAlt().length && nMismatch == 1))

  def isMNP: Boolean = getRef().length > 1 &&
    getRef().length == getAlt().length &&
    nMismatch > 1

  def isInsertion: Boolean = getRef().length < getAlt().length && getRef()(0) == getAlt()(0) && getAlt().endsWith(getRef().substring(1))

  def isDeletion: Boolean = getAlt().length < getRef().length && getRef()(0) == getAlt()(0) && getRef().endsWith(getAlt().substring(1))

  def isIndel: Boolean = isInsertion || isDeletion

  def isComplex: Boolean = getRef().length != getAlt().length && !isInsertion && !isDeletion && !isStar

  def isTransition: Boolean = isSNP && {
    val (refChar, altChar) = strippedSNP
    (refChar == 'A' && altChar == 'G') || (refChar == 'G' && altChar == 'A') ||
      (refChar == 'C' && altChar == 'T') || (refChar == 'T' && altChar == 'C')
  }

  def isTransversion: Boolean = isSNP && !isTransition

  def nMismatch: Int = {
    require(getRef().length == getAlt().length, s"invalid nMismatch call on ref `${ getRef() }' and alt `${ getAlt() }'")
    (getRef(), getAlt()).zipped.map((a, b) => if (a == b) 0 else 1).sum
  }

  def strippedSNP: (Char, Char) = {
    require(isSNP, "called strippedSNP on non-SNP")
    (getRef(), getAlt()).zipped.dropWhile { case (a, b) => a == b }.head
  }

  override def toString: String = s"${ getRef() }/${ getAlt() }"

  def compare(that: AltAlleleView): Int = {
    val c = getRef().compare(that.getRef())
    if (c != 0)
      return c

    getAlt().compare(that.getAlt())
  }
}
