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
      """assert(va.pass.forall(_.getClass.getName == "boolean"), "va.pass was not a boolean")
        |assert(va.info.AN.forall(_.getClass.getName == "int"), "AN was not an int")
        |assert(va.info.GQ_MEAN.forall(_.getClass.getName == "double"), "GQ_MEAN was not a double")
        |assert(va.info.AC.forall(_.getClass.getSimpleName == "int[]"), "AC was not an int array")
        |assert(va.filters.forall(_.getClass.getName.contains("scala.collection.immutable.Set")),
        |  "filters was not a set")
        |true""".stripMargin)).vds.expand().collect()
  }
}
