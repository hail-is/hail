package org.broadinstitute.hail.methods

import org.apache.spark.SparkException
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.Genotype
import org.testng.annotations.Test

class ImportVCFSuite extends SparkSuite {

  @Test def testInfo() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/infochar.vcf"))

    // force
    s = Count.run(s)
  }

  @Test def lineRef() {

    val line1 = "20\t10280082\t.\tA\tG\t844.69\tPASS\tAC=1;..."
    assert(LoadVCF.lineRef(line1) == "A")

    val line2 = "20\t13561632\t.\tTAA\tT\t89057.4\tPASS\tAC=2;..."
    assert(LoadVCF.lineRef(line2) == "TAA")

    assert(LoadVCF.lineRef("") == "")

    assert(LoadVCF.lineRef("this\tis\ta") == "")

    assert(LoadVCF.lineRef("20\t0\t.\t") == "")

    assert(LoadVCF.lineRef("20\t0\t.\t\t") == "")

    assert(LoadVCF.lineRef("\t\t\tabcd") == "abcd")
  }

  @Test def symbolicOrSV() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/symbolicVariant.vcf"))
    val n = s.vds.nVariants

    assert(n == 1)
    assert(VCFReport.accumulators.head._2.value(VCFReport.Symbolic) == 2)
  }

  @Test def testStoreGQ() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("--store-gq", "src/test/resources/store_gq.vcf"))

    val gqs = s.vds.flatMapWithKeys { case (v, s, g) =>
      g.gq.map { gqx => ((v.start, s), gqx) }
    }.collectAsMap()
    val expectedGQs = Map(
      (16050612, "S") -> 27,
      (16050612, "T") -> 15,
      (16051453, "S") -> 37,
      (16051453, "T") -> 52)
    assert(gqs == expectedGQs)

    s = SplitMulti.run(s, Array("--propagate-gq"))

    val f = tmpDir.createTempFile(extension = ".vcf")
    ExportVCF.run(s, Array("-o", f))

    var s2 = State(sc, sqlContext)
    s2 = ImportVCF.run(s, Array("--store-gq", "src/test/resources/store_gq_split.vcf"))

    assert(s.vds.eraseSplit.same(s2.vds))
  }

  @Test def testGlob() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    var s2 = State(sc, sqlContext)
    s2 = ImportVCF.run(s2, Array("src/test/resources/samplepart*.vcf"))

    assert(s.vds.nVariants == s2.vds.nVariants)
  }

  @Test def testUndeclaredInfo() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/undeclaredinfo.vcf"))

    assert(s.vds.vaSignature.getOption("info").isDefined)
    assert(s.vds.vaSignature.getOption("info", "undeclared").isEmpty)
    assert(s.vds.vaSignature.getOption("info", "undeclaredFlag").isEmpty)
    val infoQuerier = s.vds.vaSignature.query("info")

    val anno = s.vds
      .rdd
      .map { case (v, va, gs) => va }
      .collect()
      .head

    assert(infoQuerier(anno) != null)
  }

  @Test def testMalformed() {
    var s = State(sc, sqlContext)

    // FIXME abstract
    val e = intercept[SparkException] {
      s = ImportVCF.run(s, Array("src/test/resources/malformed.vcf"))
      s.vds.rdd.count() // force
    }
    assert(e.getMessage.contains("caught htsjdk.tribble.TribbleException$InternalCodecException: "))
  }

  @Test def testPPs() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.PPs.vcf", "--pp-as-pl"))

    val s2 = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    assert(s.vds.same(s2.vds))
  }

  @Test def testBadAD() {
    val s = ImportVCF.run(State(sc, sqlContext), Array("src/test/resources/sample_bad_AD.vcf", "--skip-bad-ad"))
    assert(s.vds.expand()
      .map(_._3)
      .collect()
      .contains(Genotype(Some(0), None, Some(30), Some(72), Some(Array(0, 72, 1080)))))

    val sFail = ImportVCF.run(State(sc, sqlContext), Array("src/test/resources/sample_bad_AD.vcf"))
    val e = intercept[SparkException] {
      sFail.vds.expand()
        .map(_._3)
        .collect()
        .contains(Genotype(Some(0), None, Some(30), Some(72), Some(Array(0, 72, 1080))))
    }
    assert(e.getMessage.contains("FatalException"))
  }
}
