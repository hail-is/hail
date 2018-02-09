package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ExplodeSuite extends SparkSuite {
  @Test def testExplode() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("foo = [1, 2, 2], bar = [1, 2, 2].toSet()")
      .annotateSamplesExpr("foo = [1, 2, 2], bar = [1, 2, 2].toSet()")
        
    assert(vsm.explodeVariants("va.foo").countVariants() == vsm.countVariants() * 3)
    assert(vsm.explodeVariants("va.bar").countVariants() == vsm.countVariants() * 2)
    assert(vsm.explodeSamples("sa.foo").numCols == vsm.numCols * 3)
    assert(vsm.explodeSamples("sa.bar").numCols == vsm.numCols * 2)

    val key = vsm.colValues(0).asInstanceOf[Row].get(1)
    val explodedKey = vsm.explodeSamples("sa.foo").colValues.take(3).asInstanceOf[IndexedSeq[Row]].map(_.get(1))
    assert(key == explodedKey)    
  }

  @Test def testExplodeUnwrap() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("foo = [1]")
      .annotateSamplesExpr("foo = [3]")
    
    val exploded = vsm.explodeVariants("va.foo")
    assert(vsm.variants.collect().sameElements(exploded.variants.collect()))

    val exploded2 = vsm.explodeSamples("sa.foo")
    assert(exploded2.stringSampleIds == vsm.stringSampleIds)
  }

  @Test def testNoElements() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("foo = NA: Array[Int32]")
      .annotateSamplesExpr("foo = NA: Array[Int32]")
    
    val exploded = vsm.explodeVariants("va.foo")
    assert(exploded.countVariants() == 0)

    val exploded2 = vsm.explodeSamples("sa.foo")
    assert(exploded2.numCols == 0)
  }
}