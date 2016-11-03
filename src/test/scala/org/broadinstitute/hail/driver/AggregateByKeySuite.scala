package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

class AggregateByKeySuite extends SparkSuite {
  @Test def replicateSampleAggregation() = {
    val inputVCF = "src/test/resources/sample.vcf"
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array(inputVCF))
    s = AnnotateSamplesExpr.run(s, Array("-c", "sa.nHet = gs.filter(g => g.isHet).count()"))
    s = AggregateByKey.run(s, Array("-k", "Sample = s", "-a", "nHet = gs.filter(g => g.isHet).count()", "-n", "kt"))

    val kt = s.ktEnv("kt")
    val vds = s.vds

    val (_, ktHetQuery) = kt.query("nHet")
    val (_, ktSampleQuery) = kt.query("Sample")
    val (_, saHetQuery) = vds.querySA("sa.nHet")

    val ktSampleResults = kt.rdd.map { case (k, v) =>
      (ktSampleQuery(k, v).map(_.asInstanceOf[String]), ktHetQuery(k, v).map(_.asInstanceOf[Long]))
    }.collectAsMap()

    assert( vds.sampleIdsAndAnnotations.forall{ case (sid, sa) => saHetQuery(sa) == ktSampleResults(Option(sid))})
  }

  @Test def replicateVariantAggregation() = {
    val inputVCF = "src/test/resources/sample.vcf"
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array(inputVCF))
    s = AnnotateVariantsExpr.run(s, Array("-c", "va.nHet = gs.filter(g => g.isHet).count()"))
    s = AggregateByKey.run(s, Array("-k", "Variant = v", "-a", "nHet = gs.filter(g => g.isHet).count()", "-n", "kt"))

    val kt = s.ktEnv("kt")
    val vds = s.vds

    val (_, ktHetQuery) = kt.query("nHet")
    val (_, ktVariantQuery) = kt.query("Variant")
    val (_, vaHetQuery) = vds.queryVA("va.nHet")

    val ktVariantResults = kt.rdd.map { case (k, v) =>
      (ktVariantQuery(k, v).map(_.asInstanceOf[Variant]), ktHetQuery(k, v).map(_.asInstanceOf[Long]))
    }.collectAsMap()

    assert( vds.variantsAndAnnotations.forall{ case (v, va) => vaHetQuery(va) == ktVariantResults(Option(v))})
  }

  @Test def replicateGlobalAggregation() = {
    val inputVCF = "src/test/resources/sample.vcf"
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array(inputVCF))
    s = AnnotateVariantsExpr.run(s, Array("-c", "va.nHet = gs.filter(g => g.isHet).count()"))
    s = AnnotateGlobalExpr.run(s, Array("-c", "global.nHet = variants.map(v => va.nHet).sum().toLong"))
    s = AggregateByKey.run(s, Array("-a", "nHet = gs.filter(g => g.isHet).count()", "-n", "kt"))

    val kt = s.ktEnv("kt")
    val vds = s.vds

    val (_, ktHetQuery) = kt.query("nHet")
    val (_, globalHetResult) = vds.queryGlobal("global.nHet")

    val ktGlobalResult = kt.rdd.map{ case (k, v) => ktHetQuery(k, v).map(_.asInstanceOf[Long])}.collect().head
    val vdsGlobalResult = globalHetResult.map(_.asInstanceOf[Long])

    assert( ktGlobalResult == vdsGlobalResult )
  }
}
