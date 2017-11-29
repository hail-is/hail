package is.hail.variant.vsm

import is.hail.SparkSuite
import org.testng.annotations.Test

class GroupBySuite extends SparkSuite {

  @Test def testGroupVariantsBy() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").annotateVariantsExpr("va.foo = gs.filter(g => g.isDefined).map(g => g.gt).sum()")
    val vds2 = vds.groupVariantsBy("va.foo", "gs.map(g => g.gt).max()", true)
    for (i <- vds2.variants.collect()) {
      println(i.asInstanceOf[Int])
    }
  }

}
