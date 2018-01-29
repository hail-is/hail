package is.hail.variant

import is.hail.annotations._
import is.hail.check._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.AltAlleleType.AltAlleleType
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

import scala.util.matching.Regex

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

object AltAlleleMethods {
  val alleleRegex: Regex = "^([ACGT]+)|\\*$".r

  def validate(allele: String) {
    if (!alleleRegex.matches(allele))
      fatal(s"""invalid allele "$allele"""")
  }

  def altAlleleType(ref: String, alt: String): AltAlleleType = {
    validate(ref)
    validate(alt)
    if (ref.length == alt.length) {
      if (ref.length == 1)
        if (ref == "*" || alt == "*")
          AltAlleleType.Star
        else
          AltAlleleType.SNP
      else {
        val mismatches = hamming(ref, alt)
        if (mismatches == 1)
          AltAlleleType.SNP
        else
          AltAlleleType.MNP
      }
    } else {
      if (ref == "*" || alt == "*")
        AltAlleleType.Star
      else if (ref.length < alt.length) {
        if (ref(0) == alt(0) && alt.endsWith(ref.substring(1)))
          AltAlleleType.Insertion
        else
          AltAlleleType.Complex
      } else {
        if (ref(0) == alt(0) && ref.endsWith(alt.substring(1)))
          AltAlleleType.Deletion
        else
          AltAlleleType.Complex
      }
    }
  }

  def isSNP(ref: String, alt: String): Boolean = altAlleleType(ref, alt) == AltAlleleType.SNP

  def isMNP(ref: String, alt: String): Boolean = altAlleleType(ref, alt) == AltAlleleType.MNP

  def isInsertion(ref: String, alt: String): Boolean = altAlleleType(ref, alt) == AltAlleleType.Insertion

  def isDeletion(ref: String, alt: String): Boolean = altAlleleType(ref, alt) == AltAlleleType.Deletion

  def isIndel(ref: String, alt: String): Boolean = {
    val aat = altAlleleType(ref, alt)
    aat == AltAlleleType.Insertion || aat == AltAlleleType.Deletion
  }

  def isStar(ref: String, alt: String): Boolean = altAlleleType(ref, alt) == AltAlleleType.Star

  def isComplex(ref: String, alt: String): Boolean = altAlleleType(ref, alt) == AltAlleleType.Complex

  def isTransition(ref: String, alt: String): Boolean = {
    val aat = altAlleleType(ref, alt)
    aat == AltAlleleType.SNP && isTransitionSNP(ref, alt)
  }

  def isTransversion(ref: String, alt: String): Boolean = {
    val aat = altAlleleType(ref, alt)
    aat == AltAlleleType.SNP && !isTransitionSNP(ref, alt)
  }

  private def isTransitionSNP(ref: String, alt: String): Boolean = {
    val pos = snpPos(ref, alt)
    val refChar = ref(pos)
    val altChar = alt(pos)
    (refChar == 'A' && altChar == 'G') || (refChar == 'G' && altChar == 'A') ||
      (refChar == 'C' && altChar == 'T') || (refChar == 'T' && altChar == 'C')
  }

  def hamming(a1: String, a2: String): Int = {
    if (a1.length != a2.length)
      fatal(s"function 'hamming' requires strings to have equal length, " +
        s"found lengths ${a1.length} and ${a2.length}: '$a1', '$a2'")
    var n = 0
    var i = 0
    while (i < a1.length) {
      if (a1(i) != a2(i))
        n += 1
      i += 1
    }
    n
  }

  def snpPos(ref: String, alt: String): Int = {
    var i = 0
    while (ref(i) == alt(i)) {
      i += 1
    }
    i
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

