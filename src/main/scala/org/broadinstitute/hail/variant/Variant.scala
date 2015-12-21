package org.broadinstitute.hail.variant

object VariantType extends Enumeration {
  type VariantType = Value
  val SNP, MNP, Insertion, Deletion, Complex = Value
}

case class Variant(contig: String,
                   // FIXME: 0- or 1-based?
                   start: Int,
                   ref: String,
                   alt: String,
                   wasSplit: Boolean = false) {
  require(ref != alt)

  import VariantType._

  def variantType: VariantType = {
    if (ref.length == 1 && alt.length == 1)
      SNP
    else if (ref.length == alt.length)
      if (nMismatch == 1)
        SNP
      else
        MNP
    else if (alt.startsWith(ref))
      Insertion
    else if (ref.startsWith(alt))
      Deletion
    else
      Complex
  }

  // PAR regions of sex chromosomes: https://en.wikipedia.org/wiki/Pseudoautosomal_region
  // Boundaries for build GRCh37: http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/
  def inXnonPAR: Boolean =
    contig == "X" && !((60001 <= start && start <= 2699520) || (154931044 <= start && start <= 155260560))
  def inYnonPAR: Boolean =
    contig == "Y" && !((10001 <= start && start <= 2649520) || (59034050 <= start && start <=  59363566))
  def isHemizygous(sex: Sex.Sex): Boolean =
    sex == Sex.Male && (inXnonPAR || inYnonPAR)


  def isSNP: Boolean = (ref.length == 1 && alt.length == 1) ||
      (ref.length == alt.length && nMismatch == 1)

  def isMNP: Boolean = ref.length > 1 &&
      ref.length == alt.length &&
      nMismatch > 1

  def isInsertion: Boolean = ref.length < alt.length && alt.startsWith(ref)

  def isDeletion: Boolean = alt.length < ref.length && ref.startsWith(alt)

  def isIndel: Boolean = isInsertion || isDeletion

  def isComplex: Boolean = ref.length != alt.length && !isInsertion && !isDeletion

  def isTransition: Boolean = isSNP && {
      val (refChar, altChar) = strippedSNP
      (refChar == 'A' && altChar == 'G') || (refChar == 'G' && altChar == 'A') ||
        (refChar == 'C' && altChar == 'T') || (refChar == 'T' && altChar == 'C')
  }

  def isTransversion: Boolean = isSNP && !isTransition

  def nMismatch: Int = {
    require(ref.length == alt.length)
    (ref,alt).zipped.map((a, b) => if (a == b) 0 else 1).sum
  }

  def strippedSNP: (Char, Char) = {
    require(isSNP)
    (ref,alt).zipped.dropWhile{ case (a, b) => a == b }.head
  }

}
