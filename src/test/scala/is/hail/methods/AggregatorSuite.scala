package is.hail.methods

import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.{VSMSubgen, VariantDataset, VariantSampleMatrix}
import is.hail.{SparkSuite, TestUtils}
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.testng.annotations.Test

class AggregatorSuite extends SparkSuite {

  @Test def testRows() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .splitMulti()
      .variantQC()
      .annotateVariantsExpr(
        """va.test.callrate = gs.fraction(g => g.isCalled()), va.test.AC = gs.map(g => g.nNonRefAlleles()).sum(),
          |va.test.AF = gs.map(g => g.nNonRefAlleles()).stats().sum.toDouble() / gs.filter(g => g.isCalled()).count() / 2.0,
          |va.test.gqstats = gs.map(g => g.gq).stats(), va.test.gqhetstats = gs.filter(g => g.isHet()).map(g => g.gq).stats(),
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
        assert(D_==(qAF(va).asInstanceOf[Double], qAFQC(va).asInstanceOf[Double]))
        assert(Option(gqStatsMean(va)).zip(Option(gqStatsMeanQC(va))).forall {
          case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
        })
        assert(Option(gqStatsStDev(va)).zip(Option(gqStatsStDevQC(va))).forall {
          case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
        })

        val gqSC = gs.aggregate(new StatCounter())({ case (s, g) =>
          if (g.isHet)
            g.gq.foreach(x => s.merge(x))
          s
        }, { case (s1, s2) => s1.merge(s2) })

        assert(Option(gqStatsHetMean(va)).zip(Option(gqSC.mean)).forall {
          case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
        })
        assert(Option(gqStatsHetStDev(va)).zip(Option(gqSC.stdev)).forall {
          case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
        })

        val lowGqGtsData = gs.filter(_.gq.exists(_ < 60))
        assert(Option(lowGqGts(va)).map(_.asInstanceOf[IndexedSeq[_]]).contains(lowGqGtsData.toIndexedSeq))

      }
  }

  @Test def testColumns() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .splitMulti()
      .sampleQC()
      .annotateSamplesExpr("sa.test.callrate = gs.fraction(g => g.isCalled()), sa.test.gqstats = " +
        "gs.map(g => g.gq).stats(), sa.test.gqhetstats = gs.filter(g => g.isHet()).map(g => g.gq).stats()")

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
          assert(Option(gqStatsMean(sa)).zip(Option(gqStatsMeanQC(sa))).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(Option(gqStatsStDev(sa)).zip(Option(gqStatsStDevQC(sa))).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(Option(gqStatsHetMean(sa)).zip(gqHetMap.get(s).map(_.mean)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(Option(gqStatsHetStDev(sa)).zip(gqHetMap.get(s).map(_.stdev)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
      }
  }

  @Test def testSum() {
    val p = Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random)) { vds =>
      val vds2 = vds.splitMulti()
        .variantQC()
        .annotateVariantsExpr("va.oneHotAC = gs.map(g => g.oneHotAlleles(v)).sum()")
        .annotateVariantsExpr("va.same = (gs.filter(g => g.isCalled()).count() == 0) || " +
          "(va.oneHotAC[0] == va.qc.nCalled * 2  - va.qc.AC) && (va.oneHotAC[1] == va.qc.nHet + 2 * va.qc.nHomVar)")
      val (_, querier) = vds2.queryVA("va.same")
      vds2.variantsAndAnnotations
        .forall { case (v, va) =>
          Option(querier(va)).exists(_.asInstanceOf[Boolean])
        }
    }
    p.check()
  }

  @Test def testMaxMin() {
    val rdd = sc.parallelize(Seq(
      Row("a",  0, null,    1,   -1, null, null,    1, null,  0l,  0f,  0d),
      Row("a", -1,   -1, null,   -2, null,    1, null, null, -1l, -1f, -1d),
      Row("a",  1,   -2,    2, null,   -1, null, null, null,  1l,  1f,  1d)), numSlices = 2)

    val signature = TStruct((("group" -> TString) +: (0 until 8).map(i => s"s$i" -> TInt))
      ++ IndexedSeq("s8" -> TLong, "s9" -> TFloat, "s10" -> TDouble): _*)

    val ktMax = new KeyTable(hc, rdd, signature)
      .aggregate("group = group", (0 until 11).map(i => s"s$i = s$i.max()").mkString(","))

    assert(ktMax.collect() == IndexedSeq(Row("a", 1, -1, 2, -1, -1, 1, 1, null, 1l, 1f, 1d)))

    val ktMin = new KeyTable(hc, rdd, signature)
      .aggregate("group = group", (0 until 11).map(i => s"s$i = s$i.min()").mkString(","))

    assert(ktMin.collect() == IndexedSeq(Row("a", -1, -2, 1, -2, -1, 1, 1, null, -1l, -1f, -1d)))
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
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Variant\\]")(
      vds.queryVariants("variants")
    )
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[String\\]")(
      vds.queryVariants("variants.map(v => v.contig)")
    )
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[Variant\\]")(
      vds.queryVariants("variants.filter(v => false)")
    )
    TestUtils.interceptFatal("unrealizable type.*Aggregable\\[String\\]")(
      vds.querySamples("samples.filter(s => false)")
    )
  }

  @Test def testCallStats() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf").cache()
      .annotateVariantsExpr(
        """va.callStats = gs.callStats(g => v),
          |va.AC = gs.map(g => g.oneHotAlleles(v)).sum(),
          |va.GC = gs.map(g => g.oneHotGenotype(v)).sum(),
          |va.AN = gs.filter(g => g.isCalled()).count() * 2""".stripMargin)
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
    assert(qTake(va) == IndexedSeq[Any](11, null, 20))
    assert(qTakeBy(va) == IndexedSeq[Any](55, null, 11))
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
      assert(vds.querySamples("samples.flatMap(g => [1,2,2].toSet()).filter(x => x % 2 == 0).sum()")._1 == 4)

      vds.annotateVariantsExpr("""va = gs.filter(g => s == "a").map(g => 1).sum()""")
        .rdd
        .collect()
        .foreach { case (_, (va, _)) => assert(va == 1) }
      true
    }.check()
  }

  @Test def testQueryGenotypes() {
    Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random)) { vds =>
      val countResult = vds.summarize().callRate
      val queryResult = vds.queryGenotypes("gs.fraction(g => g.isCalled)")._1
      val p1 = countResult.isEmpty && queryResult == null ||
        countResult.exists(x => D_==(x, queryResult.asInstanceOf[Double]))
      val filterCountResult = Some(vds.expand().count()).flatMap { r =>
        if (r == 0) None else Some(vds.expand().filter { case (v, _, g) =>
          (v.start % 2 == 1) && g.isCalled
        }.count().toDouble / r)
      }
      val queryResult2 = vds.queryGenotypes("gs.fraction(g => (v.start % 2 == 1) && g.isCalled)")._1
      val p2 = filterCountResult.isEmpty && queryResult2 == null ||
        filterCountResult.exists(x => D_==(x, queryResult2.asInstanceOf[Double]))
      p1 && p2
    }.check()
  }

  private def isLensedPrefix[T,K](lens: T => K)(prefix: Seq[T], full: Seq[T]): Boolean = {
    prefix.zip(full).forall { case (x, y) => lens(x) == lens(y) }
  }

  private def prefixModuloDisordering[T,K](sortBy: T => K)(prefix: Seq[T], full: Seq[T]): Boolean = {
    def equivClasses(ts: Seq[T]): Map[K, Set[T]] =
      ts.groupBy(sortBy).mapValues(_.toSet)

    if (prefix.isEmpty) {
      true
    } else {
      val sameOrdering = isLensedPrefix(sortBy)(prefix, full)

      val lastKey = sortBy(prefix.last)

      val prefixEquivClasses = equivClasses(prefix)
      val fullEquivClasses = equivClasses(full)

      val wholeClassesPrefix = prefixEquivClasses.filterKeys(_ != lastKey)
      val wholeClassesFull = fullEquivClasses.filterKeys(k => prefixEquivClasses.contains(k) && k != lastKey)

      val wholeClassesSame = wholeClassesFull == wholeClassesPrefix

      val lastKeySubset = prefixEquivClasses(lastKey).subsetOf(fullEquivClasses(lastKey))

      if (sameOrdering) {
        if (wholeClassesSame) {
          if (lastKeySubset) {
            true
          } else {
            println(s"The values at the last key in the prefix, $lastKey, were not a subset of those in the full list: ${prefixEquivClasses(lastKey)} ${fullEquivClasses(lastKey)}")
            false
          }
        } else {
          println(s"The values differed at some key:\n$wholeClassesPrefix\n$wholeClassesFull")
          false
        }
      } else {
        println(s"The sequences didn't have the same ordering:\n$prefix\n$full")
        false
      }
    }
  }

  @Test def takeByAndSortByAgree() {
    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)

    Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.realistic)) { (vds: VariantDataset) =>
      val Array((a, _), (b, _)) = vds.queryGenotypes(Array("gs.collect().sortBy(g => -g.gq).map(g => [g.dp, g.gq])",
        "gs.map(g => [g.dp, g.gq]).takeBy(x => x[1], 10)"))
      val sortby = a.asInstanceOf[IndexedSeq[IndexedSeq[Int]]]
      val takeby = b.asInstanceOf[IndexedSeq[IndexedSeq[Int]]]

      if (!prefixModuloDisordering((x: Seq[Int]) => x(1))(takeby, sortby)) {
        println(s"The first sequence is not a prefix, up to irrelevant disorderings, of the second sequence\n$takeby\n$sortby")
        false
      } else {
        true
      }
    } (Parameters(rng, 1000, 100))
  }

  @Test def takeByAndSortByAgreeUsingLatentEnvironment() {
    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)

    Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.realistic)) { (vds: VariantDataset) =>
      val Array((a, _), (b, _)) = vds.queryGenotypes(Array("gs.collect().sortBy(g => -g.gq).map(g => [g.dp, g.gq])",
        "gs.map(g => [g.dp, g.gq]).takeBy(x => g.gq, 10)"))
      val sortby = a.asInstanceOf[IndexedSeq[IndexedSeq[Int]]]
      val takeby = b.asInstanceOf[IndexedSeq[IndexedSeq[Int]]]

      if (!prefixModuloDisordering((x: Seq[Int]) => x(1))(takeby, sortby)) {
        println(s"The first sequence is not a prefix, up to irrelevant disorderings, of the second sequence\n$takeby\n$sortby")
        false
      } else {
        true
      }
    } (Parameters(rng, 1000, 100))
  }

}
