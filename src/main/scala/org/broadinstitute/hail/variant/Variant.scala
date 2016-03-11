package org.broadinstitute.hail.variant

import org.broadinstitute.hail.check.{Arbitrary, Gen}
import org.broadinstitute.hail.Utils._

object AltAlleleType extends Enumeration {
  type AltAlleleType = Value
  // FIXME add "*"
  val SNP, MNP, Insertion, Deletion, Complex = Value
}

object CopyState extends Enumeration {
  type CopyState = Value
  val Auto, HemiX, HemiY = Value
}

object AltAllele {
  def gen(ref: String): Gen[AltAllele] =
    for (alt <- Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))) if alt != ref)
      yield AltAllele(ref, alt)

  def gen: Gen[AltAllele] =
    for (ref <- genDNAString;
      alt <- genDNAString)
      yield AltAllele(ref, alt)
}

case class AltAllele(ref: String,
  alt: String) {
  require(ref != alt)
  require(!ref.isEmpty) // && !alt.isEmpty) FIXME

  import AltAlleleType._

  def altAlleleType: AltAlleleType = {
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
    (ref, alt).zipped.map((a, b) => if (a == b) 0 else 1).sum
  }

  def strippedSNP: (Char, Char) = {
    require(isSNP)
    (ref, alt).zipped.dropWhile { case (a, b) => a == b }.head
  }
}

object Variant {
  def apply(contig: String,
    start: Int,
    ref: String,
    alt: String): Variant = Variant(contig, start, ref, Array(AltAllele(ref, alt)))

  def nGenotypes(nAlleles: Int): Int = {
    require(nAlleles > 0)
    nAlleles * (nAlleles + 1) / 2
  }

  def compareContig(lhs: String, rhs: String): Int = {
    if (lhs.forall(_.isDigit)) {
      if (rhs.forall(_.isDigit)) {
        lhs.toInt.compare(rhs.toInt)
      } else
        -1
    } else {
      if (rhs.forall(_.isDigit))
        1
      else
        lhs.compare(rhs)
    }
  }

  def genVariants(nVariants: Int): Gen[Array[Variant]] =
    Gen.buildableOfN[Array[Variant], Variant](nVariants, gen)

  def gen: Gen[Variant] =
    for (contig <- Gen.identifier;
      start <- Gen.posInt;
      nAlleles <- Gen.frequency((5, Gen.const(2)), (1, Gen.choose(1, 10)));
      alleles <- Gen.distinctBuildableOfN[Array[String], String](
        nAlleles,
        Gen.frequency((10, genDNAString),
          (1, Gen.const("*")))) if alleles(0) != "*") yield {
      val ref = alleles(0)
      Variant(contig, start, ref, alleles.tail.map(alt => AltAllele(ref, alt)))
    }

  implicit def arbVariant: Arbitrary[Variant] = Arbitrary(gen)
}

case class Variant(contig: String,
  start: Int,
  ref: String,
  altAlleles: IndexedSeq[AltAllele]) extends Ordered[Variant] {
  /* The position is 1-based. Telomeres are indicated by using positions 0 or N+1, where N is the length of the
       corresponding chromosome or contig. See the VCF spec, v4.2, section 1.4.1. */
  require(start >= 0)
  require(!ref.isEmpty)

  def nAltAlleles: Int = altAlleles.length

  def isBiallelic: Boolean = nAltAlleles == 1

  // FIXME altAllele, alt to be deprecated
  def altAllele: AltAllele = {
    require(isBiallelic)
    altAlleles(0)
  }

  def alt: String = altAllele.alt

  def nAlleles: Int = 1 + nAltAlleles

  def alllele(i: Int): String = if (i == 0)
    ref
  else
    altAlleles(i - 1).alt

  def nGenotypes = Variant.nGenotypes(nAlleles)

  // PAR regions of sex chromosomes: https://en.wikipedia.org/wiki/Pseudoautosomal_region
  // Boundaries for build GRCh37: http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/
  def inParX: Boolean = (60001 <= start && start <= 2699520) || (154931044 <= start && start <= 155260560)

  def inParY: Boolean = (10001 <= start && start <= 2649520) || (59034050 <= start && start <= 59363566)

  import CopyState._

  def copyState(sex: Sex.Sex): CopyState =
    if (sex == Sex.Male)
      if (contig == "X" && !inParX)
        HemiX
      else if (contig == "Y" && !inParY)
        HemiY
      else
        Auto
    else
      Auto

  def compare(that: Variant): Int = {
    var c = Variant.compareContig(contig, that.contig)
    if (c != 0)
      return c

    c = start.compare(that.start)
    if (c != 0)
      return c

    c = ref.compare(that.ref)
    if (c != 0)
      return c

    c = nAltAlleles.compare(that.nAltAlleles)
    if (c != 0)
      return c

    var i = 0
    while (i < altAlleles.length) {
      c = altAlleles(i).alt.compare(that.altAlleles(i).alt)
      if (c != 0)
        return c
      i += 1
    }

    return 0
  }
}
