package is.hail.testUtils

import org.apache.spark.sql.Row
import org.json4s._

case class AltAllele(ref: String, alt: String) {
  require(ref != alt, "ref was equal to alt")
  require(!ref.isEmpty, "ref was an empty string")
  require(!alt.isEmpty, "alt was an empty string")

  import is.hail.variant.AltAlleleType._

  def altAlleleType: AltAlleleType = {
    if (isSNP)
      SNP
    else if (isInsertion)
      Insertion
    else if (isDeletion)
      Deletion
    else if (isStar)
      Star
    else if (ref.length == alt.length)
      MNP
    else
      Complex
  }

  def isStar: Boolean = alt == "*"

  def isSNP: Boolean = !isStar && ((ref.length == 1 && alt.length == 1) ||
    (ref.length == alt.length && nMismatch == 1))

  def isMNP: Boolean = ref.length > 1 &&
    ref.length == alt.length &&
    nMismatch > 1

  def isInsertion: Boolean = ref.length < alt.length && ref(0) == alt(0) && alt.endsWith(ref.substring(1))

  def isDeletion: Boolean = alt.length < ref.length && ref(0) == alt(0) && ref.endsWith(alt.substring(1))

  def isIndel: Boolean = isInsertion || isDeletion

  def isComplex: Boolean = ref.length != alt.length && !isInsertion && !isDeletion && !isStar

  def isTransition: Boolean = isSNP && {
    val (refChar, altChar) = strippedSNP
    (refChar == 'A' && altChar == 'G') || (refChar == 'G' && altChar == 'A') ||
      (refChar == 'C' && altChar == 'T') || (refChar == 'T' && altChar == 'C')
  }

  def isTransversion: Boolean = isSNP && !isTransition

  def nMismatch: Int = {
    require(ref.length == alt.length, s"invalid nMismatch call on ref `${ ref }' and alt `${ alt }'")
    (ref, alt).zipped.map((a, b) => if (a == b) 0 else 1).sum
  }

  def strippedSNP: (Char, Char) = {
    require(isSNP, "called strippedSNP on non-SNP")
    (ref, alt).zipped.dropWhile { case (a, b) => a == b }.head
  }

  def toRow: Row = Row(ref, alt)

  def toJSON: JValue = JObject(
    ("ref", JString(ref)),
    ("alt", JString(alt)))

  override def toString: String = s"$ref/$alt"

  def compare(that: AltAllele): Int = {
    val c = ref.compare(that.ref)
    if (c != 0)
      return c

    alt.compare(that.alt)
  }
}
