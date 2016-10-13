package org.broadinstitute.hail.methods

import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.{SparkSuite, TestUtils}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.check.Prop
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.broadinstitute.hail.variant.{VSMSubgen, VariantSampleMatrix}
import org.testng.annotations.Test

class AggregatorSuite extends SparkSuite {

  @Test def testRows() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    s = VariantQC.run(s, Array.empty[String])
    s = AnnotateVariants.run(s, Array("expr", "-c",
      """va.test.callrate = gs.fraction(g => g.isCalled), va.test.AC = gs.map(g => g.nNonRefAlleles).sum(),
        |va.test.AF = gs.map(g => g.nNonRefAlleles).stats().sum.toDouble / gs.filter(g => g.isCalled).count() / 2.0,
        |va.test.gqstats = gs.map(g => g.gq).stats(), va.test.gqhetstats = gs.filter(g => g.isHet).map(g => g.gq).stats(),
        |va.lowGqGts = gs.filter(g => g.gq < 60).collect()""".stripMargin))

    val qCallRate = s.vds.queryVA("va.test.callrate")._2
    val qCallRateQC = s.vds.queryVA("va.qc.callRate")._2
    val qAC = s.vds.queryVA("va.test.AC")._2
    val qACQC = s.vds.queryVA("va.qc.AC")._2
    val qAF = s.vds.queryVA("va.test.AF")._2
    val qAFQC = s.vds.queryVA("va.qc.AF")._2
    val gqStatsMean = s.vds.queryVA("va.test.gqstats.mean")._2
    val gqStatsMeanQC = s.vds.queryVA("va.qc.gqMean")._2
    val gqStatsStDev = s.vds.queryVA("va.test.gqstats.stdev")._2
    val gqStatsStDevQC = s.vds.queryVA("va.qc.gqStDev")._2
    val gqStatsHetMean = s.vds.queryVA("va.test.gqhetstats.mean")._2
    val gqStatsHetStDev = s.vds.queryVA("va.test.gqhetstats.stdev")._2
    val lowGqGts = s.vds.queryVA("va.lowGqGts")._2

    s.vds.rdd.collect()
      .foreach {
        case (v, (va, gs)) =>
          assert(qCallRate(va) == qCallRateQC(va))
          assert(qAC(va) == qACQC(va))
          assert(D_==(qAF(va).get.asInstanceOf[Double], qAFQC(va).get.asInstanceOf[Double]))
          assert(gqStatsMean(va).zip(gqStatsMeanQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsStDev(va).zip(gqStatsStDevQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })

          val gqSC = gs.aggregate(new StatCounter())({ case (s, g) =>
            if (g.isHet)
              g.gq.foreach(x => s.merge(x))
            s
          }, { case (s1, s2) => s1.merge(s2) })

          assert(gqStatsHetMean(va).zip(Option(gqSC.mean)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetStDev(va).zip(Option(gqSC.stdev)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })

          val lowGqGtsData = gs.filter(_.gq.exists(_ < 60))
          assert(lowGqGts(va).map(_.asInstanceOf[IndexedSeq[_]]).contains(lowGqGtsData.toIndexedSeq))

      }
  }

  @Test def testColumns() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])

    s = SampleQC.run(s, Array.empty[String])

    s = AnnotateSamples.run(s, Array("expr", "-c", "sa.test.callrate = gs.fraction(g => g.isCalled), sa.test.gqstats = " +
      "gs.map(g => g.gq).stats(), sa.test.gqhetstats = gs.filter(g => g.isHet).map(g => g.gq).stats()"))

    val qCallRate = s.vds.querySA("sa.test.callrate")._2
    val qCallRateQC = s.vds.querySA("sa.qc.callRate")._2
    val gqStatsMean = s.vds.querySA("sa.test.gqstats.mean")._2
    val gqStatsMeanQC = s.vds.querySA("sa.qc.gqMean")._2
    val gqStatsStDev = s.vds.querySA("sa.test.gqstats.stdev")._2
    val gqStatsStDevQC = s.vds.querySA("sa.qc.gqStDev")._2
    val gqStatsHetMean = s.vds.querySA("sa.test.gqhetstats.mean")._2
    val gqStatsHetStDev = s.vds.querySA("sa.test.gqhetstats.stdev")._2


    val gqHetMap = s.vds.aggregateBySample(new StatCounter())({ case (s, g) =>
      if (g.isHet)
        g.gq.foreach(x => s.merge(x))
      s
    }, { case (s1, s2) => s1.merge(s2) })
      .collect().toMap

    s.vds.sampleIdsAndAnnotations
      .foreach {
        case (s, sa) =>
          assert(qCallRate(sa) == qCallRateQC(sa))
          assert(gqStatsMean(sa).zip(gqStatsMeanQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsStDev(sa).zip(gqStatsStDevQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetMean(sa).zip(gqHetMap.get(s).map(_.mean)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetStDev(sa).zip(gqHetMap.get(s).map(_.stdev)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
      }
  }

  @Test def testSum() {
    val p = Prop.forAll(VariantSampleMatrix.gen(sc, VSMSubgen.random)) { vds =>
      var state = State(sc, sqlContext, vds)
      state = SplitMulti.run(state)
      state = VariantQC.run(state)
      state = AnnotateVariantsExpr.run(state, Array("-c", "va.oneHotAC = gs.map(g => g.oneHotAlleles(v)).sum()"))
      state.vds.rdd.collect()
      state = AnnotateVariantsExpr.run(state, Array("-c",
        "va.same = (gs.filter(g => g.isCalled).count() == 0) || " +
          "(va.oneHotAC[0] == va.qc.nCalled * 2  - va.qc.AC) && (va.oneHotAC[1] == va.qc.nHet + 2 * va.qc.nHomVar)"))
      val (_, querier) = state.vds.queryVA("va.same")
      state.vds.variantsAndAnnotations
        .forall { case (v, va) =>
          querier(va).exists(_.asInstanceOf[Boolean])
        }
    }
    p.check()
  }

  @Test def testHist() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf").cache()
    var s = State(sc, sqlContext, vds)

    val s2 = AnnotateVariantsExpr.run(s, Array("-c", "va = gs.map(g => g.gq).hist(0, 100, 20)"))
    s2.vds.rdd.collect.foreach { case (v, (va, gs)) =>
      val r = va.asInstanceOf[Row]

      val densities = r.getAs[IndexedSeq[Long]](1)

      val definedGq = gs.flatMap(_.gq)

      assert(densities(0) == definedGq.count(gq => gq < 5))
      assert(densities(1) == definedGq.count(gq => gq >= 5 && gq < 10))
      assert(densities.last == definedGq.count(gq => gq >= 95))
    }

    val s3 = AnnotateVariantsExpr.run(s, Array("-c", "va = gs.map(g => g.gq).hist(22, 80, 5)"))
    s3.vds.rdd.collect.foreach { case (v, (va, gs)) =>
      val r = va.asInstanceOf[Row]
      val nSmaller = r.getAs[Long](2)
      val nGreater = r.getAs[Long](3)

      val definedGq = gs.flatMap(_.gq)

      assert(nSmaller == definedGq.count(_ < 22))
      assert(nGreater == definedGq.count(_ > 80))
    }

    TestUtils.interceptFatal("""invalid bin size""") {
      AnnotateVariantsExpr.run(s, Array("-c", "va = gs.map(g => g.gq).hist(0, 0, 10)"))
    }

    TestUtils.interceptFatal("""method `hist' expects `bins' argument to be > 0""") {
      AnnotateVariantsExpr.run(s, Array("-c", "va = gs.map(g => g.gq).hist(0, 10, 0)"))
    }

    TestUtils.interceptFatal("""invalid bin size""") {
      AnnotateVariantsExpr.run(s, Array("-c", "va = gs.map(g => g.gq).hist(10, 0, 5)"))
    }

    TestUtils.interceptFatal("""method `hist' cannot contain variable references""") {
      AnnotateVariantsExpr.run(s, Array("-c", "va = gs.map(g => g.gq).hist(10, 0, va.info.AC[0])"))
    }
  }

  @Test def testErrorMessages() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf").cache()
    val s = State(sc, sqlContext, vds)

    val dummy = tmpDir.createTempFile("out")
    TestUtils.interceptFatal("""tried to export invalid type `Aggregable\[Genotype\]'""")(
      ExportVariants.run(s, Array("-o", dummy, "-c", "gs")))
    TestUtils.interceptFatal("""tried to export invalid type `Aggregable\[Int\]'""")(
      ExportVariants.run(s, Array("-o", dummy, "-c", "gs.map(g => 5)")))
    TestUtils.interceptFatal("""tried to export invalid type `Aggregable\[Genotype\]'""")(
      ExportVariants.run(s, Array("-o", dummy, "-c", "gs")))
    TestUtils.interceptFatal("""Got invalid type `Aggregable\[Genotype\]'""")(
      AnnotateVariants.run(s, Array("expr", "-c", "va = gs")))
    TestUtils.interceptFatal("""Got invalid type `Aggregable\[Int\]'""")(
      AnnotateVariants.run(s, Array("expr", "-c", "va = gs.map(G => 5)")))
    TestUtils.interceptFatal("""Got invalid type `Aggregable\[Genotype\]'""")(
      AnnotateVariants.run(s, Array("expr", "-c", "va = gs.filter(g => true)")))
  }
}
