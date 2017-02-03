package is.hail.methods

import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import is.hail.{SparkSuite, TestUtils}
import is.hail.utils._
import is.hail.check.Gen
import is.hail.check.Prop
import is.hail.driver._
import is.hail.io.vcf.LoadVCF
import is.hail.variant.{VSMSubgen, VariantSampleMatrix}
import org.testng.annotations.Test

class AggregatorSuite extends SparkSuite {

  @Test def testRows() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .splitMulti()
      .variantQC()
      .annotateVariantsExpr(
        """va.test.callrate = gs.fraction(g => g.isCalled), va.test.AC = gs.map(g => g.nNonRefAlleles).sum(),
          |va.test.AF = gs.map(g => g.nNonRefAlleles).stats().sum.toDouble / gs.filter(g => g.isCalled).count() / 2.0,
          |va.test.gqstats = gs.map(g => g.gq).stats(), va.test.gqhetstats = gs.filter(g => g.isHet).map(g => g.gq).stats(),
          |va.lowGqGts = gs.filter(g => g.gq < 60).collect()""".stripMargin)

    val qCallRate = vds.queryVA("va.test.callrate")._2
    val qCallRateQC = vds.queryVA("va.qc.callRate")._2
    val qAC = vds.queryVA("va.test.AC")._2
    val qACQC = vds.queryVA("va.qc.AC")._2
    val qAF = vds.queryVA("va.test.AF")._2
    val qAFQC = vds.queryVA("va.qc.AF")._2
    val gqStatsMean = vds.queryVA("va.test.gqstats.mean")._2
    val gqStatsMeanQC = vds.queryVA("va.qc.gqMean")._2
    val gqStatsStDev = vds.queryVA("va.test.gqstats.stdev")._2
    val gqStatsStDevQC = vds.queryVA("va.qc.gqStDev")._2
    val gqStatsHetMean = vds.queryVA("va.test.gqhetstats.mean")._2
    val gqStatsHetStDev = vds.queryVA("va.test.gqhetstats.stdev")._2
    val lowGqGts = vds.queryVA("va.lowGqGts")._2

    vds.rdd.collect()
      .foreach { case (v, (va, gs)) =>
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
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .splitMulti()
      .sampleQC()
      .annotateSamplesExpr("sa.test.callrate = gs.fraction(g => g.isCalled), sa.test.gqstats = " +
        "gs.map(g => g.gq).stats(), sa.test.gqhetstats = gs.filter(g => g.isHet).map(g => g.gq).stats()")

    val qCallRate = vds.querySA("sa.test.callrate")._2
    val qCallRateQC = vds.querySA("sa.qc.callRate")._2
    val gqStatsMean = vds.querySA("sa.test.gqstats.mean")._2
    val gqStatsMeanQC = vds.querySA("sa.qc.gqMean")._2
    val gqStatsStDev = vds.querySA("sa.test.gqstats.stdev")._2
    val gqStatsStDevQC = vds.querySA("sa.qc.gqStDev")._2
    val gqStatsHetMean = vds.querySA("sa.test.gqhetstats.mean")._2
    val gqStatsHetStDev = vds.querySA("sa.test.gqhetstats.stdev")._2

    val gqHetMap = vds.aggregateBySample(new StatCounter())({ case (s, g) =>
      if (g.isHet)
        g.gq.foreach(x => s.merge(x))
      s
    }, { case (s1, s2) => s1.merge(s2) })
      .collect().toMap

    vds.sampleIdsAndAnnotations
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
    val p = Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random)) { vds =>
      val vds2 = vds.splitMulti()
        .variantQC()
        .annotateVariantsExpr("va.oneHotAC = gs.map(g => g.oneHotAlleles(v)).sum()")
        .annotateVariantsExpr("va.same = (gs.filter(g => g.isCalled).count() == 0) || " +
          "(va.oneHotAC[0] == va.qc.nCalled * 2  - va.qc.AC) && (va.oneHotAC[1] == va.qc.nHet + 2 * va.qc.nHomVar)")
      val (_, querier) = vds2.queryVA("va.same")
      vds2.variantsAndAnnotations
        .forall { case (v, va) =>
          querier(va).exists(_.asInstanceOf[Boolean])
        }
    }
    p.check()
  }

  @Test def testHist() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf").cache()

    val vds2 = vds.annotateVariantsExpr("va = gs.map(g => g.gq).hist(0, 100, 20)")
    vds2.rdd.collect.foreach { case (v, (va, gs)) =>
      val r = va.asInstanceOf[Row]

      val frequencies = r.getAs[IndexedSeq[Long]](1)

      val definedGq = gs.flatMap(_.gq)

      assert(frequencies(0) == definedGq.count(gq => gq < 5))
      assert(frequencies(1) == definedGq.count(gq => gq >= 5 && gq < 10))
      assert(frequencies.last == definedGq.count(gq => gq >= 95))
    }

    val vds3 = vds.annotateVariantsExpr("va = gs.map(g => g.gq).hist(22, 80, 5)")
    vds3.rdd.collect.foreach { case (v, (va, gs)) =>
      val r = va.asInstanceOf[Row]
      val nLess = r.getAs[Long](2)
      val nGreater = r.getAs[Long](3)

      val definedGq = gs.flatMap(_.gq)

      assert(nLess == definedGq.count(_ < 22))
      assert(nGreater == definedGq.count(_ > 80))
    }

    TestUtils.interceptFatal("""invalid bin size""") {
      vds.annotateVariantsExpr("va = gs.map(g => g.gq).hist(0, 0, 10)")
    }

    TestUtils.interceptFatal("""method `hist' expects `bins' argument to be > 0""") {
      vds.annotateVariantsExpr("va = gs.map(g => g.gq).hist(0, 10, 0)")
    }

    TestUtils.interceptFatal("""invalid bin size""") {
      vds.annotateVariantsExpr("va = gs.map(g => g.gq).hist(10, 0, 5)")
    }

    TestUtils.interceptFatal("""symbol.*va.*not found""") {
      vds.annotateVariantsExpr("va = gs.map(g => g.gq).hist(10, 0, va.info.AC[0])")
    }
  }

  @Test def testErrorMessages() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf").cache()

    val dummy = tmpDir.createTempFile("out")
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Genotype\\]")(
      vds.exportVariants(dummy, "gs"))
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Int\\]")(
      vds.exportVariants(dummy, "gs.map(g => 5)"))
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Genotype\\]")(
      vds.exportVariants(dummy, "gs.filter(x => false)"))
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Genotype\\]")(
      vds.annotateVariantsExpr("va = gs"))
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Int\\]")(
      vds.annotateVariantsExpr("va = gs.map(g => 5)"))
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Genotype\\]")(
      vds.annotateVariantsExpr("va = gs.filter(g => true)"))
  }

  @Test def testCallStats() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf").cache()
      .annotateVariantsExpr(
        """va.callStats = gs.callStats(g => v),
          |va.AC = gs.map(g => g.oneHotAlleles(v)).sum(),
          |va.GC = gs.map(g => g.oneHotGenotype(v)).sum(),
          |va.AN = gs.filter(g => g.isCalled).count() * 2""".stripMargin)
      .annotateVariantsExpr("va.AF = va.AC / va.AN")
    val (_, csAC) = vds.queryVA("va.callStats.AC")
    val (_, csAF) = vds.queryVA("va.callStats.AF")
    val (_, csAN) = vds.queryVA("va.callStats.AN")
    val (_, csGC) = vds.queryVA("va.callStats.GC")
    val (_, ac) = vds.queryVA("va.AC")
    val (_, af) = vds.queryVA("va.AF")
    val (_, an) = vds.queryVA("va.AN")
    val (_, gc) = vds.queryVA("va.GC")

    vds.variantsAndAnnotations
      .collect()
      .foreach { case (_, va) =>
        assert(csAC(va) == ac(va), s"AC was different")
        assert(csAN(va) == an(va), s"AN was different")
        assert(csAF(va) == af(va), s"AF was different")
        assert(csGC(va) == gc(va), s"GC was different")
      }
  }

  @Test def testCounter() {
    Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.plinkSafeBiallelic)) { vds =>
      val (r, t) = vds.queryVariants("variants.map(v => v.contig).counter()")
      val counterMap = r.asInstanceOf[Map[String, Long]]
      val aggMap = vds.variants.map(_.contig).countByValue()
      aggMap == counterMap
    }.check()
  }

  @Test def testTake() {
    val vds = hc.importVCF("src/test/resources/aggTake.vcf")
      .annotateVariantsExpr("va.take = gs.map(g => g.dp).take(3)")
      .annotateVariantsExpr("va.takeBy = gs.map(g => g.dp).takeBy(dp => g.gq, 3)")

    val (_, qTake) = vds.queryVA("va.take")
    val (_, qTakeBy) = vds.queryVA("va.takeBy")

    val va = vds.variantsAndAnnotations.map(_._2).collect().head
    assert(qTake(va).contains(IndexedSeq[Any](11, null, 20)))
    assert(qTakeBy(va).contains(IndexedSeq[Any](55, null, 11)))
  }

  @Test def testTransformations() {
    Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random.copy(sampleIdGen = Gen.const(Array("a", "b"))))) { vds =>
      assert(vds.querySamples("samples.map(id => if (id == \"b\") (NA : Sample) else id).map(x => 1).sum()")._1 == 2)
      assert(vds.querySamples("samples.filter(id => true).map(id => 1).sum()")._1 == 2)
      assert(vds.querySamples("samples.filter(id => false).map(id => 1).sum()")._1 == 0)
      assert(vds.querySamples("samples.flatMap(g => [1]).sum()")._1 == 2)
      assert(vds.querySamples("samples.flatMap(g => [0][:0]).sum()")._1 == 0)
      assert(vds.querySamples("samples.flatMap(g => [1,2]).sum()")._1 == 6)
      assert(vds.querySamples("samples.flatMap(g => [1,2]).filter(x => x % 2 == 0).sum()")._1 == 4)
      assert(vds.querySamples("samples.flatMap(g => [1,2,2].toSet).filter(x => x % 2 == 0).sum()")._1 == 4)

      vds.annotateVariantsExpr("""va = gs.filter(g => s.id == "a").map(g => 1).sum()""")
        .rdd
        .collect()
        .foreach { case (_, (va, _)) => assert(va == 1) }
      true
    }.check()
  }
}
