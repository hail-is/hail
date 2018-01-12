package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ExplodeSuite extends SparkSuite {
  @Test def testExplode() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.foo = [1, 2, 2], va.bar = [1, 2, 2].toSet()")
      .annotateSamplesExpr("sa.foo = [1, 2, 2], sa.bar = [1, 2, 2].toSet()")
        
    assert(vsm.explodeVariants("va.foo").countVariants() == vsm.countVariants() * 3)
    assert(vsm.explodeVariants("va.bar").countVariants() == vsm.countVariants() * 2)
    assert(vsm.explodeSamples("sa.foo").nSamples == vsm.nSamples * 3)
    assert(vsm.explodeSamples("sa.bar").nSamples == vsm.nSamples * 2)
    
    val key = vsm.sampleAnnotations(0).asInstanceOf[Row].get(0)
    val explodedKey = vsm.explodeSamples("sa.foo").sampleAnnotations.take(3).asInstanceOf[IndexedSeq[Row]].map(_.get(0))       
    assert(key == explodedKey)    
  }

  @Test def testExplodeUnwrap() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.foo = [1]")
      .annotateSamplesExpr("sa.foo = [3]")
    
    val exploded = vsm.explodeVariants("va.foo")
    assert(vsm.variants.collect().sameElements(exploded.variants.collect()))

    val exploded2 = vsm.explodeSamples("sa.foo")
    assert(exploded2.sampleIds == vsm.sampleIds)
  }

  @Test def testNoElements() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.foo = NA: Array[Int32]")
      .annotateSamplesExpr("sa.foo = NA: Array[Int32]")
    
    val exploded = vsm.explodeVariants("va.foo")
    assert(exploded.countVariants() == 0)

    val exploded2 = vsm.explodeSamples("sa.foo")
    assert(exploded2.nSamples == 0)
  }
}