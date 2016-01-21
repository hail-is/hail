package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class LinkageDisiquilibriumSuite extends SparkSuite {
  @Test def test() {
    val variant_list = vds.rdd.map(a => (a._1, a._3)).toLocalIterator.toList
    val vds = LoadVCF(sc, "src/test/resources/linkageDisiquilibrium.vcf")
    val pruned = LinkageDisequilibrium.Prune(variant_list, 0.045, 0.1).toList
    assert(pruned.size == 6)
    assert(variant_list(0) == pruned(0))
    assert(variant_list(3) == pruned(1))
    assert(variant_list(4) == pruned(2))
    assert(variant_list(6) == pruned(3))
    assert(variant_list(7) == pruned(4))
    assert(variant_list(8) == pruned(5))
  }
}
