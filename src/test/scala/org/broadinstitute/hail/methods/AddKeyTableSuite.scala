package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._

class AddKeyTableSuite extends SparkSuite {
  @Test def test1() {
    var s = State(sc, sqlContext, null)
    s = ImportVCF.run(s, Array("-i", "src/test/resources/sample.vcf"))
    s = AnnotateVariantsExpr.run(s, Array("-c", "va.foo = gs.filter(g => g.isHet).count()"))
    s = AnnotateSamplesExpr.run(s, Array("-c", "sa.foo = gs.filter(g => g.isHet).count()"))
    s = AnnotateGlobalExpr.run(s, Array("-c", "global.foo = variants.count()"))
    s = PrintSchema.run(s, Array.empty[String])
    s = AddKeyTable.run(s, Array("-k", "foo = va.foo, foo1 = global.foo, foo2 = sa.foo"))
  }
}
