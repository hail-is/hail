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

    FilterGenotypes.run(state, Array("--keep", "-c", "g.isHet"))
      .vds.expand().collect()
      .foreach { case (v, s, g) => assert(g.isNotCalled || g.isHet) }

    FilterGenotypes.run(state, Array("--remove", "-c", "g.call.map(c => c.gq < 20).getOrElse(false)"))
      .vds.expand().collect()
      .foreach { case (v, s, g) => g.call.foreach(c => c.gq >= 20) }
  }
}
