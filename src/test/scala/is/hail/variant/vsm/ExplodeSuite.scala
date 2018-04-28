package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ExplodeSuite extends SparkSuite {
  @Test def testExplode() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateRowsExpr("foo" -> "[1, 2, 2]", "bar" -> "[1, 2, 2].toSet()")
      .annotateColsExpr("foo" -> "[1, 2, 2]", "bar" -> "[1, 2, 2].toSet()")
        
    assert(vsm.explodeRows("va.foo").countRows() == vsm.countRows() * 3)
    assert(vsm.explodeRows("va.bar").countRows() == vsm.countRows() * 2)
    assert(vsm.explodeCols("sa.foo").numCols == vsm.numCols * 3)
    assert(vsm.explodeCols("sa.bar").numCols == vsm.numCols * 2)

    val key = vsm.colValues.value(0).asInstanceOf[Row].get(1)
    val explodedKey = vsm.explodeCols("sa.foo").colValues.value.take(3).asInstanceOf[IndexedSeq[Row]].map(_.get(1))
    assert(key == explodedKey)    
  }

  @Test def testExplodeUnwrap() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateRowsExpr("foo" -> "[1]")
      .annotateColsExpr("foo" -> "[3]")
    
    val exploded = vsm.explodeRows("va.foo")
    assert(vsm.variants.collect().sameElements(exploded.variants.collect()))

    val exploded2 = vsm.explodeCols("sa.foo")
    assert(exploded2.stringSampleIds == vsm.stringSampleIds)
  }

  @Test def testNoElements() {
    val vsm = hc.importVCF("src/test/resources/sample.vcf")
      .annotateRowsExpr("foo" -> "NA: Array[Int32]")
      .annotateColsExpr("foo" -> "NA: Array[Int32]")
    
    val exploded = vsm.explodeRows("va.foo")
    assert(exploded.countRows() == 0)

    val exploded2 = vsm.explodeCols("sa.foo")
    assert(exploded2.numCols == 0)
  }
}