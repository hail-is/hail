package org.broadinstitute.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class AltAlleleSuite extends TestNGSuite {
  @Test def testAltAllele() {

    val ti = AltAllele("A", "G")
    val tv = AltAllele("A", "T")
    val snp1 = AltAllele("AT", "AC")
    val snp2 = AltAllele("CT", "GT")
    val mnp1 = AltAllele("CA", "TT")
    val mnp2 = AltAllele("ACTGAC", "ATTGTT")
    val insertion = AltAllele("A", "ATGC")
    val deletion = AltAllele("ATGC", "A")
    val complex1 = AltAllele("CTA", "ATTT")
    val complex2 = AltAllele("A", "TATGC")

    assert(ti.nMismatch == 1 &&
      snp1.nMismatch == 1 &&
      mnp2.nMismatch == 3)

    assert(ti.strippedSNP == ('A', 'G') &&
      snp1.strippedSNP == ('T', 'C') &&
      snp2.strippedSNP == ('C', 'G'))

    assert(ti.isTransition && tv.isTransversion &&
      snp1.isTransition && snp2.isTransversion)
    assert(!ti.isTransversion && !tv.isTransition &&
      !snp1.isTransversion && !snp2.isTransition)

    assert(ti.isSNP && tv.isSNP && snp1.isSNP && snp2.isSNP)
    assert(!mnp1.isSNP && !mnp2.isSNP &&
      !insertion.isSNP && !deletion.isSNP &&
      !complex1.isSNP && !complex2.isSNP)

    assert(mnp1.isMNP && mnp2.isMNP)
    assert(!snp1.isMNP && !snp2.isMNP &&
      !insertion.isMNP && !deletion.isMNP &&
      !complex1.isMNP && !complex2.isMNP)

    assert(insertion.isInsertion && deletion.isDeletion)
    assert(!snp1.isInsertion && !snp2.isDeletion &&
      !mnp1.isInsertion && !mnp2.isDeletion &&
      !insertion.isDeletion && !deletion.isInsertion &&
      !complex1.isInsertion && !complex2.isDeletion)

    assert(insertion.isIndel && deletion.isIndel)
    assert(!snp1.isIndel && !snp2.isIndel &&
      !mnp1.isIndel && !mnp2.isIndel &&
      !complex1.isIndel && !complex2.isIndel)

    assert(complex1.isComplex && complex2.isComplex)
    assert(!snp1.isComplex && !snp2.isComplex &&
      !mnp1.isComplex && !mnp2.isComplex &&
      !insertion.isComplex && !deletion.isComplex)

    import AltAlleleType._

    val altAlleles = Array(ti, tv, snp1, snp2, mnp1, mnp2,
      insertion, deletion, complex1, complex2)

    // FIXME: use ScalaCheck
    for (a <- altAlleles) {
      assert((a.altAlleleType == SNP) == a.isSNP)
      assert((a.altAlleleType == MNP) == a.isMNP)
      assert((a.altAlleleType == Insertion) == a.isInsertion)
      assert((a.altAlleleType == Deletion) == a.isDeletion)
      assert((a.altAlleleType == Complex) == a.isComplex)
    }
  }
}
