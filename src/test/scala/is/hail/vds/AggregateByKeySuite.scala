package is.hail.vds

import is.hail.SparkSuite
import is.hail.expr.TLong
import is.hail.utils._
import is.hail.variant._
import org.testng.annotations.Test

class AggregateByKeySuite extends SparkSuite {
  @Test def replicateSampleAggregation() {
    val inputVCF = "src/test/resources/sample.vcf"
    val vds = hc.importVCF(inputVCF)
      .annotateSamplesExpr("sa.nHet = gs.filter(g => g.isHet).count()")

    val kt = vds.aggregateByKey("Sample = s", "nHet = g.map(g => g.isHet.toInt).sum()")

    val (_, ktHetQuery) = kt.query("nHet")
    val (_, ktSampleQuery) = kt.query("Sample")
    val (_, saHetQuery) = vds.querySA("sa.nHet")

    val ktSampleResults = kt.rdd.map { case (k, v) =>
      (ktSampleQuery(k, v).map(_.asInstanceOf[String]), ktHetQuery(k, v).map(_.asInstanceOf[Int]))
    }.collectAsMap()

    assert(vds.sampleIdsAndAnnotations.forall { case (sid, sa) => saHetQuery(sa) == ktSampleResults(Option(sid)) })
  }

  @Test def replicateVariantAggregation() {
    val inputVCF = "src/test/resources/sample.vcf"
    val vds = hc.importVCF(inputVCF)
      .annotateVariantsExpr("va.nHet = gs.filter(g => g.isHet).count()")

    val kt = vds.aggregateByKey("Variant = v", "nHet = g.map(g => g.isHet.toInt).sum()")

    val (_, ktHetQuery) = kt.query("nHet")
    val (_, ktVariantQuery) = kt.query("Variant")
    val (_, vaHetQuery) = vds.queryVA("va.nHet")

    val ktVariantResults = kt.rdd.map { case (k, v) =>
      (ktVariantQuery(k, v).map(_.asInstanceOf[Variant]), ktHetQuery(k, v).map(_.asInstanceOf[Int]))
    }.collectAsMap()

    assert(vds.variantsAndAnnotations.forall { case (v, va) => vaHetQuery(va) == ktVariantResults(Option(v)) })
  }

  @Test def replicateGlobalAggregation() {
    val inputVCF = "src/test/resources/sample.vcf"
    var vds = hc.importVCF(inputVCF)
      .annotateVariantsExpr("va.nHet = gs.filter(g => g.isHet).count().toInt")

    vds = vds.annotateGlobal(vds.queryVariants("variants.map(v => va.nHet).sum()")._1, TLong, "global.nHet")
    val kt = vds.aggregateByKey("", "nHet = g.map(g => g.isHet.toInt).sum()")

    val (_, ktHetQuery) = kt.query("nHet")
    val (_, globalHetResult) = vds.queryGlobal("global.nHet")

    val ktGlobalResult = kt.rdd.map { case (k, v) => ktHetQuery(k, v).map(_.asInstanceOf[Int]) }.collect().head
    val vdsGlobalResult = globalHetResult.map(_.asInstanceOf[Int])

    assert(ktGlobalResult == vdsGlobalResult)
  }
}
