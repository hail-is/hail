package org.broadinstitute.hail.methods

import java.io.File
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.Main._
import org.broadinstitute.hail.driver.{FilterVariants, FilterSamples, FilterGenotypes, State}
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)

    assert(FilterSamples.run(state, Array("--keep", "-c", "\"^HG\" ~ s.id"))
      .vds.nLocalSamples == 63)

    assert(FilterVariants.run(state, Array("--remove", "-c", "v.start >= 14066228"))
      .vds.nVariants == 173)

    val highGQ = FilterGenotypes.run(state, Array("--remove", "-c", "g.call.exists(c => c.gq < 20)"))
      .vds.expand().collect()

    assert(!highGQ.exists { case (v, s, g) => g.call.exists(c => c.gq < 20) })
    assert(highGQ.count{ case (v, s, g) => g.call.exists(c => c.gq >= 20) } == 31260)

    val vds2 = TestRDDBuilder.buildRDD(1, 1, sc)
    val state2 = State("", sc, sqlContext, vds2)
    val nVariants = vds2.nVariants

    assert(FilterVariants.run(state2, Array("--remove", "-c", "Array(1,2).size === 2"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "true === true"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "true"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "5 === 5"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "5.0 === 5.0"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "5 == 5.0"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "5.0 === 5"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "5 === 5.0"))
      .vds.nVariants == 0)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "val a = new FilterOption[Int](Some(4)); val b = new FilterOption[Int](Some(5)); a > b"))
      .vds.nVariants == nVariants)

    assert(FilterVariants.run(state2, Array("--remove", "-c", "new FilterOption[Int](Some(4)) > 5"))
      .vds.nVariants == nVariants)

  }
}
