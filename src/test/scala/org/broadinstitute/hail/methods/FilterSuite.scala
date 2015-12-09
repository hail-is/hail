package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
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

    val highGQ = FilterGenotypes.run(state, Array("--remove", "-c", "g.call.exists(c => c.gq < 20)"))
      .vds.expand().collect()

    assert(!highGQ.exists { case (v, s, g) => g.call.exists(c => c.gq < 20) })
    assert(highGQ.count{ case (v, s, g) => g.call.exists(c => c.gq >= 20) } == 31260)

    // the below command will test typing of runtime-generated code exposing annotations
    FilterGenotypes.run(state, Array("--keep", "-c",
      """assert(va.pass.getClass.getName == "boolean");""" +
        """assert(va.info.AN.getClass.getName == "int");""" +
        """assert(va.info.GQ_MEAN.getClass.getName == "double");""" +
        """assert(va.info.AC.getClass.getName == "int[]");""" +
        """assert(va.filters.getClass.getName.contains("scala.collection.immutable.Set"));true"""))
  }
}
