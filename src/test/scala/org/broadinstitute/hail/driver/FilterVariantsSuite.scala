package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.{Gen, Prop}
import org.testng.annotations.Test

class FilterVariantsSuite extends SparkSuite {
  @Test def test() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/sample2.vcf"))

    val variants = s.vds.variants.collect().toSet

    val f = tmpDir.createTempFile("test", extension = ".variant_list")
    Prop.check(forAll(Gen.subset(variants), Gen.arbBoolean) { case (subset, keep) =>
      writeTextFile(f, s.hadoopConf) { s =>
        for (v <- subset) {
          s.write(v.toString)
          s.write("\n")
        }
      }

      val t = FilterVariantsList.run(s, Array(
        if (keep)
          "--keep"
        else
          "--remove",
        "-i", f))

      val tVariants = t.vds.variants.collect().toSet
      if (keep)
        tVariants == subset
      else
        (tVariants.union(subset) == variants
          && tVariants.intersect(subset).isEmpty)
    })
  }

  @Test def testFilterAll() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/sample2.vcf"))
    s = FilterVariantsAll.run(s)

    assert(s.vds.nVariants == 0)
  }
}
