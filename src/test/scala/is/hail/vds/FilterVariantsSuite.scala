package is.hail.vds

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.testUtils._
import is.hail.check.Arbitrary._
import is.hail.check.Prop._
import is.hail.check.{Gen, Prop}
import org.testng.annotations.Test

class FilterVariantsSuite extends SparkSuite {
  @Test def test() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf", nPartitions = Some(8))
    assert(vds.nPartitions == 8)

    val variants = vds.variants.collect().toSet

    assert(vds.filterVariantsList(variants, keep = true).countVariants() == variants.size)
    assert(vds.filterVariantsList(variants, keep = false).countVariants() == 0)

    assert(vds.filterVariantsList(Set.empty[Annotation], keep = false).countVariants() == variants.size)
    assert(vds.filterVariantsList(Set.empty[Annotation], keep = true).countVariants() == 0)

    Prop.check(forAll(Gen.subset(variants), arbitrary[Boolean]) { case (subset, keep) =>
      val filtered = vds.filterVariantsList(subset, keep)

      val filteredVariants = filtered.variants.collect().toSet
      if (keep)
        filteredVariants == subset
      else
        variants -- subset == filteredVariants
    })
  }

  @Test def testFilterAll() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .dropVariants()

    assert(vds.countVariants == 0)
  }
}
