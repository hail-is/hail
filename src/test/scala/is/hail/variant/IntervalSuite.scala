package is.hail.variant

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.{Gen, Prop}
import is.hail.expr.types._
import is.hail.io.annotators.IntervalList
import is.hail.utils._
import is.hail.testUtils._
import org.testng.annotations.Test

class IntervalSuite extends SparkSuite {
  val gr = GenomeReference.GRCh37
  val pord = gr.locusType.ordering

  def genomicInterval(contig: String, start: Int, end: Int): Interval =
    Interval(Locus(contig, start), Locus(contig, end), true, false)

  @Test def test() {
    val ilist = IntervalTree.apply(pord, Array(
      genomicInterval("1", 10, 20),
      genomicInterval("1", 30, 40),
      genomicInterval("2", 40, 50)))

    assert(!ilist.contains(pord, Locus("1", 5)))
    assert(ilist.contains(pord, Locus("1", 10)))
    assert(ilist.contains(pord, Locus("1", 15)))
    assert(!ilist.contains(pord, Locus("1", 20)))
    assert(!ilist.contains(pord, Locus("1", 25)))
    assert(ilist.contains(pord, Locus("1", 35)))

    assert(!ilist.contains(pord, Locus("2", 30)))
    assert(ilist.contains(pord, Locus("2", 45)))

    assert(!ilist.contains(pord, Locus("3", 1)))

    val ex1 = IntervalList.read(hc, "src/test/resources/example1.interval_list")

    val f = tmpDir.createTempFile("example", extension = "interval_list")

    ex1.annotate("""interval = "[" + interval.start.contig + ":" + interval.start.position + "-" + interval.end.position + "]"""")
      .export(f)

    val ex1wr = hc.importTable(f).annotate("interval = LocusInterval(interval)").keyBy("interval")

    assert(ex1wr.same(ex1))

    val ex2 = IntervalList.read(hc, "src/test/resources/example2.interval_list")
    assert(ex1.select(Array("interval")).same(ex2))
  }

  @Test def testAll() {
    val vds = MatrixTable.fromLegacy[Annotation](hc,
      MatrixType.fromParts(
        globalType = TStruct.empty(),
        colKey = Array("s"),
        colType = TStruct("s" -> TString()),
        rowPartitionKey = Array("v"), rowKey = Array("v"),
        rowType = TStruct("v" -> TVariant(GenomeReference.defaultReference)),
        entryType = Genotype.htsGenotypeType),
      Annotation.empty, IndexedSeq.empty[Annotation],
      sc.parallelize(Seq((Annotation(Variant("1", 100, "A", "T")), Iterable.empty[Annotation]))))

    val intervalFile = tmpDir.createTempFile("intervals")
    hadoopConf.writeTextFile(intervalFile) { out =>
      out.write("1\t50\t150\t+\tTHING1\n")
      out.write("1\t50\t150\t+\tTHING2\n")
      out.write("1\t50\t150\t+\tTHING3\n")
      out.write("1\t50\t150\t+\tTHING4\n")
      out.write("1\t50\t150\t+\tTHING5")
    }

    assert(vds.annotateVariantsTable(IntervalList.read(hc, intervalFile), "foo", product = true)
      .annotateVariantsExpr("foo = va.foo.map(x => x.target)")
      .rowsTable().query("foo.collect()[0].toSet()")._1 == Set("THING1", "THING2", "THING3", "THING4", "THING5"))
  }

  @Test def testNew() {
    val a = Array(
      genomicInterval("1", 10, 20),
      genomicInterval("1", 30, 40),
      genomicInterval("2", 40, 50))

    val t = IntervalTree.apply(pord, a)

    assert(t.contains(pord, Locus("1", 15)))
    assert(t.contains(pord, Locus("1", 30)))
    assert(!t.contains(pord, Locus("1", 40)))
    assert(!t.contains(pord, Locus("1", 20)))
    assert(!t.contains(pord, Locus("1", 25)))
    assert(!t.contains(pord, Locus("1", 9)))
    assert(!t.contains(pord, Locus("5", 20)))

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

    val t2 = IntervalTree.annotationTree(pord, a2.map(i => (i, ())))

    assert(t2.queryIntervals(pord, Locus("1", 30)).toSet == Set(
      genomicInterval("1", 30, 40), genomicInterval("1", 30, 50),
      genomicInterval("1", 29, 31), genomicInterval("1", 30, 31)))

  }

  @Test def properties() {

    // greater chance of collision
    val lgen = Gen.zip(Gen.oneOf("1", "2"), Gen.choose(1, 100))
      .map { case (c, p) => Locus(c, p) }
    val g = Gen.zip(IntervalTree.gen(pord, lgen),
      lgen)

    val p = Prop.forAll(g) { case (it, locus) =>
      val intervals = it.map(_._1).toSet

      val setResults = intervals.filter(_.contains(pord, locus))
      val treeResults = it.queryIntervals(pord, locus).toSet

      val inSet = intervals.exists(_.contains(pord, locus))
      val inTree = it.contains(pord, locus)

      setResults == treeResults &&
        setResults.nonEmpty == inSet &&
        inSet == inTree
    }
    p.check()
  }

  @Test def testAnnotateIntervalsAll() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .annotateVariantsTable(IntervalList.read(hc, "src/test/resources/annotinterall.interval_list"),
        "annot", product = true)
      .annotateVariantsExpr("annot = va.annot.map(x => x.target)")

    val (t, q) = vds.queryVA("va.annot")
    assert(t == TArray(TString()))

    vds.variantRDD.foreach { case (v1, (va, gs)) =>
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
  
  @Test def testParser() {
    val xMax = gr.contigLength("X")
    val yMax = gr.contigLength("Y")
    val chr22Max = gr.contigLength("22")

    assert(Locus.parseInterval("[1:100-1:101)", gr) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("[1:100-101)", gr) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("[X:100-101)", gr) == Interval(Locus("X", 100), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:100-end)", gr) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:100-End)", gr) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:100-END)", gr) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:start-101)", gr) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:Start-101)", gr) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:START-101)", gr) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:START-Y:END)", gr) == Interval(Locus("X", 1), Locus("Y", yMax), true, false))
    assert(Locus.parseInterval("[X-Y)", gr) == Interval(Locus("X", 1), Locus("Y", yMax), true, false))
    assert(Locus.parseInterval("[1-22)", gr) == Interval(Locus("1", 1), Locus("22", chr22Max), true, false))

    assert(Locus.parseInterval("[16:29500000-30200000)", gr) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[16:29.5M-30.2M)", gr) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[16:29500K-30200K)", gr) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[1:100K-2:200K)", gr) == Interval(Locus("1", 100000), Locus("2", 200000), true, false))

    assert(Locus.parseInterval("[1:1.111K-2000)", gr) == Interval(Locus("1", 1111), Locus("1", 2000), true, false))
    assert(Locus.parseInterval("[1:1.111111M-2000000)", gr) == Interval(Locus("1", 1111111), Locus("1", 2000000), true, false))

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

    val gr37 = GenomeReference.GRCh37
    val gr38 = GenomeReference.GRCh38

    val x = "[GL000197.1:3739-GL000202.1:7538)"
    assert(Locus.parseInterval(x, gr37) ==
      Interval(Locus("GL000197.1", 3739), Locus("GL000202.1", 7538), true, false))

    val y = "[HLA-DRB1*13:02:01:5-HLA-DRB1*14:05:01:100)"
    assert(Locus.parseInterval(y, gr38) ==
      Interval(Locus("HLA-DRB1*13:02:01", 5), Locus("HLA-DRB1*14:05:01", 100), true, false))

    val z = "[HLA-DRB1*13:02:01:5-100)"
    assert(Locus.parseInterval(z, gr38) ==
      Interval(Locus("HLA-DRB1*13:02:01", 5), Locus("HLA-DRB1*13:02:01", 100), true, false))
  }

  @Test def testEndpoints() {
    val ord = TInt32().ordering
    val intContains = (i: Interval, v: Int) => i.contains(ord, v)

    for (s1 <- Array(true, false); e1 <- Array(true, false)) {

      val i0 = Interval(1, 1, includeStart=s1, includeEnd=e1)

      assert(i0.isEmpty(ord) == (!i0.includeStart || !i0.includeEnd))

      val i1 = Interval(1, 5, includeStart=s1, includeEnd=e1)

      assert(i1.contains(ord, 3))
      assert(i1.includeStart == i1.contains(ord, 1))
      assert(i1.includeEnd == i1.contains(ord, 5))

      val iLocus = Interval(Locus("1", 100), Locus("1", 101), s1, e1)
      assert(Locus.parseInterval(iLocus.toString, gr) == iLocus)

      for (s2 <- Array(true, false); e2 <- Array(true, false)) {
        val i2 = Interval(5, 10, includeStart=s2, includeEnd=e2)
        assert(i1.overlaps(ord, i2) == i2.overlaps(ord, i1))
        assert(i1.overlaps(ord, i2) == (i1.includeEnd && i2.includeStart))

        val iTree = IntervalTree.apply(ord, Array(i1, i2))
        assert(iTree.contains(ord, 5) == (i1.includeEnd || i2.includeStart))
        assert(iTree.queryIntervals(ord, 5).length == (if (i1.includeEnd || i2.includeStart) 1 else 0))
        assert(iTree.contains(ord, 1) == i1.includeStart)
        assert(iTree.contains(ord, 10) == i2.includeEnd)

        val i3 = Interval(2, 3, includeStart=s2, includeEnd=e2)
        assert(i1.overlaps(ord, i3))
        assert(i3.overlaps(ord, i1))

        val i4 = Interval(1, 5, includeStart=s2, includeEnd=e2)
        assert((Interval.ordering(ord).compare(i1, i4) < 0) == ((i1.includeStart && !i4.includeStart) || ((i1.includeStart == i4.includeStart) && i4.includeEnd && !i1.includeEnd)))
        assert((Interval.ordering(ord).compare(i1, i4) == 0) == ((i1.includeStart == i4.includeStart) && (i4.includeEnd == i1.includeEnd)))
        assert((Interval.ordering(ord).compare(i1, i4) > 0) == ((!i1.includeStart && i4.includeStart) || ((i1.includeStart == i4.includeStart) && !i4.includeEnd && i1.includeEnd)))
      }
    }

    val itree = IntervalTree.apply(ord, Array(
      Interval(1, 5, false, false),
      Interval(1, 5, true, true),
      Interval(10, 20, false, false),
      Interval(10, 20, true, true)))
    val itree2 = IntervalTree.apply(ord, Array(
      Interval(1, 5, true, true),
      Interval(10, 20, true, true)))
    val pruned = new ArrayBuilder[Interval]()
    val pruned2 = new ArrayBuilder[Interval]()
    itree.foreach { case (interval, _) => pruned += interval}
    itree2.foreach { case (interval, _) => pruned2 += interval}

    assert(pruned.size == pruned2.size)
    assert(pruned.result().zip(pruned2.result()).forall { case (i1, i2) => i1 == i2 })
  }
}
