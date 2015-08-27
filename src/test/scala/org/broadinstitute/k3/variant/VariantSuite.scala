package org.broadinstitute.k3.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class VariantSuite extends TestNGSuite {
  @Test def testVariant() {

    val ti = Variant("chr1", 1234, "A", "G")
    val tv = Variant("chr1", 1234, "A", "T")
    val snp1 = Variant("chr1", 1234, "AT", "AC")
    val snp2 = Variant("chr1", 1234, "CT", "GT")
    val mnp1 = Variant("chr1", 1234, "CA", "TT")
    val mnp2 = Variant("chr1", 1234, "ACTGAC", "ATTGTT")
    val insertion = Variant("chr1", 1234, "A", "ATGC")
    val deletion = Variant("chr1", 1234, "ATGC", "A")
    val complex1 = Variant("chr1", 1234, "CTA", "ATTT")
    val complex2 = Variant("chr1", 1234, "A", "TATGC")

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

    import VariantType._

    val variants = Array(ti, tv, snp1, snp2, mnp1, mnp2,
      insertion, deletion, complex1, complex2)

    // FIXME: use ScalaCheck
    for (v <- variants) {
      assert((v.variantType == SNP) == v.isSNP)
      assert((v.variantType == MNP) == v.isMNP)
      assert((v.variantType == Insertion) == v.isInsertion)
      assert((v.variantType == Deletion) == v.isDeletion)
      assert((v.variantType == Complex) == v.isComplex)
    }
  }
}
