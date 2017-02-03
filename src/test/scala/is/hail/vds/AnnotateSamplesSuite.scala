package is.hail.vds

import is.hail.SparkSuite
import is.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

class AnnotateSamplesSuite extends SparkSuite {

  @Test def testVDS() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")


    val selfAnnotated = vds.annotateSamplesVDS(vds, root = Some("sa.other"))

    val (_, q) = selfAnnotated.querySA("sa.other")
    assert(vds.sampleIdsAndAnnotations == selfAnnotated.sampleIdsAndAnnotations.map { case (id, anno) =>
      (id, q(anno).orNull)
    })
  }
}
