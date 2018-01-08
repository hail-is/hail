package is.hail.variant

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.{Gen, Prop}
import is.hail.expr.typ._
import is.hail.io.annotators.IntervalList
import is.hail.utils._
import org.testng.annotations.Test

class IntervalSuite extends SparkSuite {
  val gr = GenomeReference.GRCh37
  implicit val locusOrd: Ordering[Locus] = gr.locusOrdering

  def genomicInterval(contig: String, start: Int, end: Int): Interval[Locus] =
    Interval(Locus(contig, start), Locus(contig, end))

  @Test def test() {
    val ilist = IntervalTree.apply[Locus](Array(
      genomicInterval("1", 10, 20),
      genomicInterval("1", 30, 40),
      genomicInterval("2", 40, 50)))

    assert(!ilist.contains(Locus("1", 5)))
    assert(ilist.contains(Locus("1", 10)))
    assert(ilist.contains(Locus("1", 15)))
    assert(!ilist.contains(Locus("1", 20)))
    assert(!ilist.contains(Locus("1", 25)))
    assert(ilist.contains(Locus("1", 35)))

    assert(!ilist.contains(Locus("2", 30)))
    assert(ilist.contains(Locus("2", 45)))

    assert(!ilist.contains(Locus("3", 1)))

    val ex1 = IntervalList.read(hc, "src/test/resources/example1.interval_list")

    val f = tmpDir.createTempFile("example", extension = ".interval_list")

    ex1.annotate("""interval = interval.start.contig + ":" + interval.start.position + "-" + interval.end.position""")
      .export(f)

    val ex1wr = hc.importTable(f).annotate("interval = LocusInterval(interval)").keyBy("interval")

    assert(ex1wr.same(ex1))

    val ex2 = IntervalList.read(hc, "src/test/resources/example2.interval_list")
    assert(ex1.select(Array("interval")).same(ex2))
  }

  @Test def testAll() {
    val vds = MatrixTable.fromLegacy(hc, VSMFileMetadata(Array.empty[String]),
      sc.parallelize(Seq((Variant("1", 100, "A", "T"), (Annotation.empty, Iterable.empty[Annotation])))))

    val intervalFile = tmpDir.createTempFile("intervals")
    hadoopConf.writeTextFile(intervalFile) { out =>
      out.write("1\t50\t150\t+\tTHING1\n")
      out.write("1\t50\t150\t+\tTHING2\n")
      out.write("1\t50\t150\t+\tTHING3\n")
      out.write("1\t50\t150\t+\tTHING4\n")
      out.write("1\t50\t150\t+\tTHING5")
    }

    assert(vds.annotateVariantsTable(IntervalList.read(hc, intervalFile), root = "va", product = true)
      .variantsAndAnnotations
      .collect()
      .head._2.asInstanceOf[IndexedSeq[_]].toSet == Set("THING1", "THING2", "THING3", "THING4", "THING5"))
  }

  @Test def testNew() {
    val a = Array(
      genomicInterval("1", 10, 20),
      genomicInterval("1", 30, 40),
      genomicInterval("2", 40, 50))

    val t = IntervalTree.apply[Locus](a)

    assert(t.contains(Locus("1", 15)))
    assert(t.contains(Locus("1", 30)))
    assert(!t.contains(Locus("1", 40)))
    assert(!t.contains(Locus("1", 20)))
    assert(!t.contains(Locus("1", 25)))
    assert(!t.contains(Locus("1", 9)))
    assert(!t.contains(Locus("5", 20)))

    // Test queries
    val a2 = Array(
      genomicInterval("1", 10, 20),
      genomicInterval("1", 30, 40),
      genomicInterval("1", 30, 50),
      genomicInterval("1", 29, 31),
      genomicInterval("1", 29, 30),
      genomicInterval("1", 30, 31),
      genomicInterval("2", 41, 50),
      genomicInterval("2", 43, 50),
      genomicInterval("2", 45, 70),
      genomicInterval("2", 42, 43),
      genomicInterval("2", 1, 10))

    val t2 = IntervalTree.annotationTree(a2.map(i => (i, ())))

    assert(t2.queryIntervals(Locus("1", 30)).toSet == Set(
      genomicInterval("1", 30, 40), genomicInterval("1", 30, 50),
      genomicInterval("1", 29, 31), genomicInterval("1", 30, 31)))

  }

  @Test def properties() {

    // greater chance of collision
    val lgen = Gen.zip(Gen.oneOf("1", "2"),
      Gen.choose(1, 100))
    val g = Gen.zip(IntervalTree.gen(lgen),
      lgen)

    val p = Prop.forAll(g) { case (it, locus) =>
      val intervals = it.map(_._1).toSet

      val setResults = intervals.filter(_.contains(locus))
      val treeResults = it.queryIntervals(locus).toSet

      val inSet = intervals.exists(_.contains(locus))
      val inTree = it.contains(locus)

      setResults == treeResults &&
        setResults.nonEmpty == inSet &&
        inSet == inTree
    }
    p.check()
  }

  @Test def testAnnotateIntervalsAll() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .annotateVariantsTable(IntervalList.read(hc, "src/test/resources/annotinterall.interval_list"),
        root = "va.annot",
        product = true)

    val (t, q) = vds.queryVA("va.annot")
    assert(t == TArray(TString()))

    vds.rdd.foreach { case (v1, (va, gs)) =>
      val v = v1.asInstanceOf[Variant]
      val a = q(va).asInstanceOf[IndexedSeq[String]].toSet

      if (v.start == 17348324)
        simpleAssert(a == Set("A", "B"))
      else if (v.start >= 17333902 && v.start <= 17370919)
        simpleAssert(a == Set("A"))
      else
        simpleAssert(a == Set())
    }
  }

  @Test def testFilter() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf", nPartitions = Some(4)).cache()
    val iList = tmpDir.createTempFile("input", ".interval_list")
    val tmp1 = tmpDir.createTempFile("output", ".tsv")

    val startPos = 16050036 - 250000
    val endPos = 17421565 + 250000
    val intervalGen = for (start <- Gen.choose(startPos, endPos);
      end <- Gen.choose(start, endPos))
      yield Interval(Locus("22", start), Locus("22", end))
    val intervalsGen = for (nIntervals <- Gen.choose(0, 10);
      g <- Gen.buildableOfN[Array](nIntervals, intervalGen)) yield g

    Prop.forAll(intervalsGen.filter(_.nonEmpty)) { intervals =>
      hadoopConf.writeTextFile(iList) { out =>
        intervals.foreach { i =>
          out.write(s"22\t${ i.start.position }\t${ i.end.position }\n")
        }
      }

      val vdsKeep = vds.filterVariantsTable(IntervalList.read(hc, iList), keep = true)
      val vdsRemove = vds.filterVariantsTable(IntervalList.read(hc, iList), keep = false)

      val p1 = vdsKeep.same(vds.copy(rdd = vds.rdd.filter { case (v, _) =>
          intervals.exists(_.contains(v.asInstanceOf[Variant].locus))
      }))

      val p2 = vdsRemove.same(vds.copy(rdd = vds.rdd.filter { case (v, _) =>
        intervals.forall(!_.contains(v.asInstanceOf[Variant].locus))
      }))

      val p = p1 && p2
      if (!p)
        println(
          s"""ASSERTION FAILED
            |  p1: $p1
            |  p2: $p2""".stripMargin)
      p
    }.check()
  }

  @Test def testParser() {
    assert(Locus.parseInterval("1:100-1:101", gr) == Interval(Locus("1", 100), Locus("1", 101)))
    assert(Locus.parseInterval("1:100-101", gr) == Interval(Locus("1", 100), Locus("1", 101)))
    assert(Locus.parseInterval("X:100-101", gr) == Interval(Locus("X", 100), Locus("X", 101)))
    assert(Locus.parseInterval("X:100-end", gr) == Interval(Locus("X", 100), Locus("X", Int.MaxValue)))
    assert(Locus.parseInterval("X:100-End", gr) == Interval(Locus("X", 100), Locus("X", Int.MaxValue)))
    assert(Locus.parseInterval("X:100-END", gr) == Interval(Locus("X", 100), Locus("X", Int.MaxValue)))
    assert(Locus.parseInterval("X:start-101", gr) == Interval(Locus("X", 0), Locus("X", 101)))
    assert(Locus.parseInterval("X:Start-101", gr) == Interval(Locus("X", 0), Locus("X", 101)))
    assert(Locus.parseInterval("X:START-101", gr) == Interval(Locus("X", 0), Locus("X", 101)))
    assert(Locus.parseInterval("X:START-Y:END", gr) == Interval(Locus("X", 0), Locus("Y", Int.MaxValue)))
    assert(Locus.parseInterval("X-Y", gr) == Interval(Locus("X", 0), Locus("Y", Int.MaxValue)))
    assert(Locus.parseInterval("1-22", gr) == Interval(Locus("1", 0), Locus("22", Int.MaxValue)))

    assert(Locus.parseInterval("16:29500000-30200000", gr) == Interval(Locus("16", 29500000), Locus("16", 30200000)))
    assert(Locus.parseInterval("16:29.5M-30.2M", gr) == Interval(Locus("16", 29500000), Locus("16", 30200000)))
    assert(Locus.parseInterval("16:29500K-30200K", gr) == Interval(Locus("16", 29500000), Locus("16", 30200000)))
    assert(Locus.parseInterval("1:100K-2:200K", gr) == Interval(Locus("1", 100000), Locus("2", 200000)))

    assert(Locus.parseInterval("1:1.111K-2000", gr) == Interval(Locus("1", 1111), Locus("1", 2000)))
    assert(Locus.parseInterval("1:1.111111M-2000000", gr) == Interval(Locus("1", 1111111), Locus("1", 2000000)))

    intercept[IllegalArgumentException] {
      Locus.parseInterval("X:101-100", gr)
    }

    intercept[IllegalArgumentException] {
      Locus.parseInterval("4:start-3:end", gr)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("4::start-5:end", gr)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("4:start-", gr)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("1:1.1111K-2k", gr)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("1:1.1111111M-2M", gr)
    }
  }
}
