package org.broadinstitute.k3.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class VariantSuite extends TestNGSuite {
  @Test def testVariant() {
    val tv = Variant("chr1", 1234, "A", "T")
    val ti = Variant("chr1", 1234, "A", "G")

    val insertion = Variant("chr1", 1234, "A", "ATGC")
    val deletion = Variant("chr1", 1234, "ATGC", "A")

    assert(tv.isSNP && ti.isSNP)
    assert(!tv.isTransition && tv.isTransversion)
    assert(ti.isTransition && !ti.isTransversion)
    assert(insertion.isInsertion)
    assert(deletion.isDeletion)

    assert(!ti.isComplex && !tv.isComplex && !insertion.isComplex && !deletion.isComplex)
  }
}
