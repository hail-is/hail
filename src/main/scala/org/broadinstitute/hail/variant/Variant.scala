package org.broadinstitute.hail.variant


object VariantType extends Enumeration {
  type VariantType = Value
  val SNP, MNP, Insertion, Deletion, Complex = Value
}

object CopyState extends Enumeration {
  type CopyState = Value
  val Auto, HemiX, HemiY = Value
}


case class Variant(contig: String,
                   // FIXME: 0- or 1-based?
                   start: Int,
                   ref: String,
                   alt: String) extends Ordered[Variant]{
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
  def inParX: Boolean = (60001 <= start && start <= 2699520) || (154931044 <= start && start <= 155260560)
  def inParY: Boolean = (10001 <= start && start <= 2649520) || ( 59034050 <= start && start <=  59363566)

  import CopyState._

  def ploidy(sex: Sex.Sex): CopyState =
    if (sex == Sex.Male)
      if (contig == "X" && !inParX)
        HemiX
      else if (contig == "Y" && !inParY)
        HemiY
      else
        Auto
    else
      Auto

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

  def compare(that:Variant):Int = {
    if (this.contig != that.contig)
      this.contig.compare(that.contig)
    else if (this.start != that.start)
      this.start.compare(that.start)
    else if (this.ref != that.ref)
      this.ref.compare(that.ref)
    else
      this.alt.compare(that.alt)
  }
}
