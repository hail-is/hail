package is.hail.vds

import is.hail.SparkSuite
import is.hail.check.Arbitrary._
import is.hail.check.Prop._
import is.hail.check.{Gen, Prop}
import is.hail.utils._
import org.testng.annotations.Test

class FilterVariantsSuite extends SparkSuite {
  @Test def test() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")

    val variants = vds.variants.collect().toSet

    val f = tmpDir.createTempFile("test", extension = ".variant_list")
    Prop.check(forAll(Gen.subset(variants), arbitrary[Boolean]) { case (subset, keep) =>
      hadoopConf.writeTextFile(f) { s =>
        for (v <- subset) {
          s.write(v.toString)
          s.write("\n")
        }
      }

      val filtered = vds.filterVariantsList(f, keep)

      val filteredVariants = filtered.variants.collect().toSet
      if (keep)
        filteredVariants == subset
      else
        (filteredVariants.union(subset) == variants
          && filteredVariants.intersect(subset).isEmpty)
    })
  }

  @Test def testFilterAll() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .dropVariants()

    assert(vds.countVariants == 0)
  }
}
