package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.TestUtils
import org.testng.annotations.Test

class ExplodeSuite extends SparkSuite {
  @Test def testExplode() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.foo = [1, 2, 3]")
    val exploded = vsm.explodeVariants("va.foo")
    assert(exploded.countVariants() == vsm.countVariants()*3)

    val vsm2 = hc.importVCF("src/test/resources/sample.vcf")
      .annotateSamplesExpr("sa.foo = [1, 2, 3]")
    val exploded2 = vsm2.explodeSamples("sa.foo")
    assert(exploded2.nSamples == vsm.nSamples*3)
  }

  @Test def testExplodeUnwrap() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.foo = [1]")
    val exploded = vsm.explodeVariants("va.foo")
    assert(vsm.variants.collect().sameElements(exploded.variants.collect()))

    val vsm2 = hc.importVCF("src/test/resources/sample.vcf")
      .annotateSamplesExpr("sa.foo = [3]")
    val exploded2 = vsm2.explodeSamples("sa.foo")
    assert(exploded2.sampleIds.sameElements(vsm.sampleIds))
  }

  @Test def testNoElements() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.foo = []")
    val exploded = vsm.explodeVariants("va.foo")
    assert(vsm.countVariants() == 0)

    val vsm2 = hc.importVCF("src/test/resources/sample.vcf")
      .annotateSamplesExpr("sa.foo = []")
    val exploded2 = vsm2.explodeSamples("sa.foo")
    assert(exploded2.nSamples == 0)
  }
}