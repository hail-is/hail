package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {

  @Test def filterTest() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State(sc, sqlContext, vds.cache())

    assert(FilterSamples.run(state, Array("--keep", "-c", "\"^HG\" ~ s.id"))
      .vds.nLocalSamples == 63)

    assert(FilterVariants.run(state, Array("--remove", "-c", "v.start >= 14066228"))
      .vds.nVariants == 173)

    assert(FilterVariants.run(state, Array("--keep", "-c", "va.pass"))
      .vds.nVariants == 312)

    assert(FilterVariants.run(state, Array("--keep", "-c", "va.info.AN == 200"))
      .vds.nVariants == 310)

    assert(FilterVariants.run(state, Array("--keep", "-c", "va.info.AC.contains(20)"))
      .vds.nVariants == 3)

    assert(FilterVariants.run(state, Array("--keep", "-c", """va.filters.contains("VQSRTrancheSNP99.60to99.80")"""))
      .vds.nVariants == 3)

    // FIXME: rsid of "." should be treated as missing value
    assert(FilterVariants.run(state, Array("--keep", "-c", """va.rsid != ".""""))
      .vds.nVariants == 258)

    assert(FilterVariants.run(state, Array("--remove", "-c", """va.rsid == ".""""))
      .vds.nVariants == 258)

    val stateWithSampleQC = SampleQC.run(state, Array("--store"))

    assert(FilterSamples.run(stateWithSampleQC, Array("--keep", "-c", "sa.qc.nCalled == 337"))
      .vds.nLocalSamples == 17)

    assert(FilterSamples.run(stateWithSampleQC, Array("--keep", "-c", "sa.qc.dpMean > 60"))
      .vds.nLocalSamples == 7)

    assert(FilterSamples.run(stateWithSampleQC, Array("--keep", "-c", "if (\"^C1048\" ~ s.id) {sa.qc.rTiTv > 3.5 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > 3"))
      .vds.nLocalSamples == 14)

    val stateWithVariantQC = VariantQC.run(state, Array("--store"))

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.nCalled < 100"))
      .vds.nVariants == 36)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.nHomVar > 0 && va.qc.nHet > 0"))
      .vds.nVariants == 104)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar > 0"))
      .vds.nVariants == 104)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar >= 0"))
      .vds.nVariants == 117)

    assert(FilterVariants.run(stateWithVariantQC, Array("--remove", "-c", "va.qc.rHetHomVar.isMissing"))
      .vds.nVariants == 117)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar.isNotMissing"))
      .vds.nVariants == 117)

    val highGQ = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20"))
      .vds.expand().collect()

    assert(!highGQ.exists { case (v, s, g) => g.call.exists(c => c.gq < 20) })
    assert(highGQ.count{ case (v, s, g) => g.call.exists(c => c.gq >= 20) } == 31260)

    val highGQorMidQGAndLowFS = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20 || (g.gq < 30 && va.info.FS > 30)"))
      .vds.expand().collect()

    val vds2 = LoadVCF(sc, "src/test/resources/sample_filter.vcf")
    val state2 = State(sc, sqlContext, vds2.cache())

    assert(FilterGenotypes.run(state2, Array("--keep", "-c", "g.ad(0) < 30")).vds.expand().collect().count(_._3.isCalled) == 3)

    assert(FilterGenotypes.run(state2, Array("--keep", "-c", "g.ad(1).toDouble / g.dp > 0.05")).vds.expand().collect().count(_._3.isCalled) == 3)

    val highGQ2 = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20"))

    assert(!highGQ2.vds.expand().collect().exists { case (v, s, g) => g.call.exists(c => c.gq < 20) })

    val chr1 = FilterVariants.run(state2, Array("--keep", "-c", "v.contig == \"1\"" ))

    assert(chr1.vds.rdd.count == 9)

    assert(chr1.vds.expand().collect().count(_._3.isCalled) == 9 * 11 - 2)

    val hetOrHomVarOnChr1 = FilterGenotypes.run(chr1, Array("--remove", "-c", "g.isHomRef"))
      .vds.expand().collect()

    assert(hetOrHomVarOnChr1.count(_._3.isCalled) == 9 + 3 + 3) // remove does not retain the 2 missing genotypes

    val homRefOnChr1 = FilterGenotypes.run(chr1, Array("--keep", "-c", "g.isHomRef"))
      .vds.expand().collect()

    assert(homRefOnChr1.count(_._3.isCalled) == 9 * 11 - (9 + 3 + 3) - 2) // keep does not retain the 2 missing genotypes

  }

  @Test def filterFromFileTest() {

    val vds = TestRDDBuilder.buildRDD(8, 8, sc)

    val state = State(sc, sqlContext, vds)

    assert(FilterSamples.run(state, Array("--keep", "-c", "src/test/resources/filter.sample_list")).vds.nLocalSamples == 3)

    assert(FilterSamples.run(state, Array("--remove", "-c", "src/test/resources/filter.sample_list")).vds.nLocalSamples == 5)

    assert(FilterVariants.run(state, Array("--keep", "-c", "src/test/resources/filter.interval_list")).vds.nVariants == 6)

    assert(FilterVariants.run(state, Array("--remove", "-c", "src/test/resources/filter.interval_list")).vds.nVariants == 2)

  }

}
