package is.hail.vds

import is.hail.SparkSuite
import is.hail.expr.TInt
import is.hail.utils._
import is.hail.variant._
import org.testng.annotations.Test

class AggregateByKeySuite extends SparkSuite {
  @Test def replicateSampleAggregation() {
    val inputVCF = "src/test/resources/sample.vcf"
    val vds = hc.importVCF(inputVCF)
      .annotateSamplesExpr("sa.nHet = gs.filter(g => g.isHet()).count()")

    val kt = vds.aggregateByKey("Sample = s", "nHet = g.map(g => g.isHet().toInt()).sum()")

    val (_, ktHetQuerier) = kt.queryRow("nHet")
    val (_, ktSampleQuerier) = kt.queryRow("Sample")
    val (_, saHetQuerier) = vds.querySA("sa.nHet")

    val ktSampleResults = kt.rdd.map { a =>
      (Option(ktSampleQuerier(a)), Option(ktHetQuerier(a)).map(_.asInstanceOf[Int]))
    }.collectAsMap()

    assert(vds.sampleIdsAndAnnotations.forall { case (sid, sa) => Option(saHetQuerier(sa)) == ktSampleResults(Option(sid)) })
  }

  @Test def replicateVariantAggregation() {
    val inputVCF = "src/test/resources/sample.vcf"
    val vds = hc.importVCF(inputVCF)
      .annotateVariantsExpr("va.nHet = gs.filter(g => g.isHet()).count()")

    val kt = vds.aggregateByKey("Variant = v", "nHet = g.map(g => g.isHet().toInt()).sum()")

    val (_, ktHetQuerier) = kt.queryRow("nHet")
    val (_, ktVariantQuerier) = kt.queryRow("Variant")
    val (_, vaHetQuerier) = vds.queryVA("va.nHet")

    val ktVariantResults = kt.rdd.map { a =>
      (Option(ktVariantQuerier(a)).map(_.asInstanceOf[Variant]), Option(ktHetQuerier(a)).map(_.asInstanceOf[Int]))
    }.collectAsMap()

    assert(vds.variantsAndAnnotations.forall { case (v, va) => Option(vaHetQuerier(va)) == ktVariantResults(Option(v)) })
  }

  @Test def replicateGlobalAggregation() {
    val inputVCF = "src/test/resources/sample.vcf"
    var vds = hc.importVCF(inputVCF)
      .annotateVariantsExpr("va.nHet = gs.filter(g => g.isHet()).count().toInt()")
    
    vds = vds.annotateGlobal(vds.queryVariants("variants.map(v => va.nHet).sum()")._1, TInt, "global.nHet")
    val kt = vds.aggregateByKey("", "nHet = g.map(g => g.isHet.toInt).sum()")

    val (_, ktHetQuerier) = kt.queryRow("nHet")
    val (_, globalHetResult) = vds.queryGlobal("global.nHet")

    val ktGlobalResult = kt.rdd.map { a => Option(ktHetQuerier(a)).map(_.asInstanceOf[Int]) }.collect().head
    val vdsGlobalResult = Option(globalHetResult).map(_.asInstanceOf[Int])

    assert(ktGlobalResult == vdsGlobalResult)
  }
}
