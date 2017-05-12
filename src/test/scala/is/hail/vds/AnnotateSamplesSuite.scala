package is.hail.vds

import is.hail.SparkSuite
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class AnnotateSamplesSuite extends SparkSuite {
  @Test def testKeyTable() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true).keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/sample2.vcf")

    val sampleMap = vds.annotateSamplesTable(kt, expr = "sa.annot = table")
      .querySamples("index(samples.map(s => {s: s, anno: sa.annot}).collect(), s).mapValues(x => x.anno)")._1
      .asInstanceOf[Map[_, _]]

    val ktMap = kt.keyedRDD()
      .map { case (k, v) => (k.getAs[String](0), v) }
      .collectAsMap()

    assert(sampleMap == ktMap)

    // should run without throwing an index error
    vds.annotateSamplesTable(kt.filter("false", keep = true), expr = "sa.annot = table")

    val sampleMap2 = vds.annotateSamplesTable(kt, vdsKey = List("s"), expr = "sa.annot = table")
      .querySamples("index(samples.map(s => {s: s, anno: sa.annot}).collect(), s).mapValues(x => x.anno)")._1
      .asInstanceOf[Map[_, _]]

    assert(sampleMap2 == ktMap)
  }
}
