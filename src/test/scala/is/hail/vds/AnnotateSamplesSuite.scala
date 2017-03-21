package is.hail.vds

import is.hail.SparkSuite
import is.hail.utils.TextTableConfiguration
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class AnnotateSamplesSuite extends SparkSuite {

  @Test def testVDS() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")

    val selfAnnotated = vds.annotateSamplesVDS(vds, root = Some("sa.other"))

    val (_, q) = selfAnnotated.querySA("sa.other")
    assert(vds.sampleIdsAndAnnotations == selfAnnotated.sampleIdsAndAnnotations.map { case (id, anno) =>
      (id, q(anno))
    })
  }

  @Test def testKeyTable() {
    val kt = hc.importKeyTable(List("src/test/resources/sampleAnnotations.tsv"),
      keys = List("Sample"),
      config = TextTableConfiguration(impute = true))


    val vds = hc.importVCF("src/test/resources/sample2.vcf")

    val sampleMap = vds.annotateSamplesKeyTable(kt, "sa.annot = table")
      .querySamples("index(samples.map(s => {s: s.id, anno: sa.annot}).collect(), s).mapValues(x => x.anno)")._1
      .asInstanceOf[Map[_, _]]

    val ktMap = kt.rdd
      .map { r => (r.asInstanceOf[Row].getAs[String](0), r) }
      .collectAsMap()

    assert(sampleMap == ktMap)


    val sampleMap2 = vds.annotateSamplesKeyTable(kt, List("s.id"), "sa.annot = table")
      .querySamples("index(samples.map(s => {s: s.id, anno: sa.annot}).collect(), s).mapValues(x => x.anno)")._1
      .asInstanceOf[Map[_, _]]

    assert(sampleMap2 == ktMap)
  }
}
