package org.broadinstitute.hail.methods

import java.io.File
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.Main._
import org.broadinstitute.hail.driver.{FilterVariants, FilterSamples, FilterGenotypes, State}
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)

    assert(FilterSamples.run(state, Array("--keep", "-c", "\"^HG\" ~ s.id"))
      .vds.nLocalSamples == 63)

    assert(FilterVariants.run(state, Array("--remove", "-c", "v.start >= 14066228"))
      .vds.nVariants == 173)

    val highGQ = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq.exists(_ < 20)"))
      .vds.expand().collect()

    assert(!highGQ.exists { case (v, s, g) => g.gq.exists(_ < 20) })
    assert(highGQ.count{ case (v, s, g) => g.gq.exists(_ >= 20) } == 30889)
  }
}
