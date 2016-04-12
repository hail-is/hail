package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.check.Prop._
import org.testng.annotations.Test

class FilterVariantsSuite extends SparkSuite {
  @Test def test() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/sample2.vcf"))

    val variants = s.vds.variants.collect().toSet

    Prop.check(forAll(Gen.subset(variants), Gen.arbBoolean) { case (subset, keep) =>
        writeTextFile("/tmp/test.variant_list", s.hadoopConf) { s =>
          for (v <- subset) {
            s.write(v.toString)
            s.write("\n")
          }
        }

        val t = FilterVariants.run(s, Array(
          if (keep)
            "--keep"
          else
            "--remove",
          "-c", "/tmp/test.variant_list"))

        val tVariants = t.vds.variants.collect().toSet
        if (keep)
          tVariants == subset
        else
          (tVariants.union(subset) == variants
            && tVariants.intersect(subset).isEmpty)
      })
  }
}
