package is.hail.variant

import is.hail.utils._

import scala.util.matching.Regex

object AltAlleleType extends Enumeration {
  type AltAlleleType = Value
  val SNP, MNP, Insertion, Deletion, Complex, Star, Symbolic, Unknown = Value
}

object CopyState extends Enumeration {
  type CopyState = Value
  val Auto, HemiX, HemiY = Value
}

object AltAlleleMethods {
  import is.hail.variant.AltAlleleType._

  val baseRegex: Regex = "^([ACGTNM])+$".r

  val symbolicRegex: Regex = "(^\\.)|(\\.$)|(^<)|(>$)|(\\[)|(\\])".r

  def altAlleleType(ref: String, alt: String): AltAlleleType = {
    if (!baseRegex.matches(ref))
      AltAlleleType.Unknown
    else if (baseRegex.matches(alt))
      if (ref.length == alt.length) {
        if (ref.length == 1)
          AltAlleleType.SNP
        else {
          val mismatches = hamming(ref, alt)
          if (mismatches == 1)
            AltAlleleType.SNP
          else
            AltAlleleType.MNP
        }
      } else {
        if (ref.length < alt.length) {
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
      } else if (alt == "*")
      AltAlleleType.Star
    else if (symbolicRegex.matches(alt))
      AltAlleleType.Symbolic
    else
      AltAlleleType.Unknown
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
