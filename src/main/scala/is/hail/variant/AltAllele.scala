package is.hail.variant

import is.hail.annotations._
import is.hail.check._
import is.hail.expr.typ._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

object AltAlleleType extends Enumeration {
  type AltAlleleType = Value
  val SNP, MNP, Insertion, Deletion, Complex, Star = Value
}

object CopyState extends Enumeration {
  type CopyState = Value
  val Auto, HemiX, HemiY = Value
}

object AltAllele {

  def fromRegionValue(m: Region, offset: Long): AltAllele = {
    val t = TAltAllele.representation()
    val ref = TString.loadString(m, t.loadField(m, offset, 0))
    val alt = TString.loadString(m, t.loadField(m, offset, 1))
    AltAllele(ref, alt)
  }

  def sparkSchema: StructType = StructType(Array(
    StructField("ref", StringType, nullable = false),
    StructField("alt", StringType, nullable = false)))

  def fromRow(r: Row): AltAllele =
    AltAllele(r.getString(0), r.getString(1))

  def gen(ref: String): Gen[AltAllele] =
    for (alt <- Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))) if alt != ref)
      yield AltAllele(ref, alt)

  def gen: Gen[AltAllele] =
    for (ref <- genDNAString;
      alt <- genDNAString if alt != ref)
      yield AltAllele(ref, alt)

  implicit def altAlleleOrder: Ordering[AltAllele] = new Ordering[AltAllele] {
    def compare(x: AltAllele, y: AltAllele): Int = x.compare(y)
  }
}

trait IAltAllele {
  def ref(): String

  def alt(): String

  import AltAlleleType._

  def reify(): AltAllele =
    AltAllele(ref, alt)

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

  def isInsertion: Boolean = ref.length < alt.length && ref()(0) == alt()(0) && alt.endsWith(ref.substring(1))

  def isDeletion: Boolean = alt.length < ref.length && ref()(0) == alt()(0) && ref.endsWith(alt.substring(1))

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

case class AltAllele(ref: String, alt: String) extends IAltAllele {
  require(ref != alt, "ref was equal to alt")
  require(!ref.isEmpty, "ref was an empty string")
  require(!alt.isEmpty, "alt was an empty string")

  override def reify(): AltAllele = this
}

