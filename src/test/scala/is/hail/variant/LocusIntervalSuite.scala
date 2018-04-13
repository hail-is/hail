package is.hail.variant

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.{Gen, Prop}
import is.hail.expr.types._
import is.hail.io.annotators.IntervalList
import is.hail.methods.FilterIntervals
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LocusIntervalSuite extends SparkSuite {
  val rg = ReferenceGenome.GRCh37
  val pord = rg.locusType.ordering

  def genomicInterval(contig: String, start: Int, end: Int): Interval =
    Interval(Locus(contig, start), Locus(contig, end), true, false)

  @Test def testAnnotateVariantsTable() {
    val vds = MatrixTable.fromLegacy[Annotation](hc,
      MatrixType.fromParts(
        globalType = TStruct.empty(),
        colKey = Array("s"),
        colType = TStruct("s" -> TString()),
        rowPartitionKey = Array("locus"), rowKey = Array("locus", "alleles"),
        rowType = TStruct(
          "locus" -> TLocus(ReferenceGenome.defaultReference),
          "alleles" -> TArray(TString())),
        entryType = Genotype.htsGenotypeType),
      Annotation.empty, IndexedSeq.empty[Annotation],
      sc.parallelize(Seq((Annotation(Locus("1", 100), Array("A", "T").toFastIndexedSeq), Iterable.empty[Annotation]))))

    val intervalFile = tmpDir.createTempFile("intervals")
    hadoopConf.writeTextFile(intervalFile) { out =>
      out.write("1\t50\t150\t+\tTHING1\n")
      out.write("1\t50\t150\t+\tTHING2\n")
      out.write("1\t50\t150\t+\tTHING3\n")
      out.write("1\t50\t150\t+\tTHING4\n")
      out.write("1\t50\t150\t+\tTHING5")
    }

    assert(vds.annotateRowsTable(IntervalList.read(hc, intervalFile), "foo", product = true)
      .annotateRowsExpr("foo" -> "va.foo.map(x => x.target)")
      .rowsTable().aggregate("AGG.map(r => r.foo).collect()[0].toSet()")._1 == Set("THING1", "THING2", "THING3", "THING4", "THING5"))
  }

  @Test def testAnnotateIntervalsAll() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .annotateRowsTable(IntervalList.read(hc, "src/test/resources/annotinterall.interval_list"),
        "annot", product = true)
      .annotateRowsExpr("annot" -> "va.annot.map(x => x.target)")

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
    val xMax = rg.contigLength("X")
    val yMax = rg.contigLength("Y")
    val chr22Max = rg.contigLength("22")

    assert(Locus.parseInterval("[1:100-1:101)", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("[1:100-101)", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("[X:100-101)", rg) == Interval(Locus("X", 100), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:100-end)", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:100-End)", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:100-END)", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:start-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:Start-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:START-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:START-Y:END)", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, false))
    assert(Locus.parseInterval("[X-Y)", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, false))
    assert(Locus.parseInterval("[1-22)", rg) == Interval(Locus("1", 1), Locus("22", chr22Max), true, false))

    assert(Locus.parseInterval("1:100-1:101", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("1:100-101", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("X:100-end", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, true))
    assert(Locus.parseInterval("(X:100-End]", rg) == Interval(Locus("X", 100), Locus("X", xMax), false, true))
    assert(Locus.parseInterval("(X:100-END)", rg) == Interval(Locus("X", 100), Locus("X", xMax), false, false))
    assert(Locus.parseInterval("[X:start-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("(X:Start-101]", rg) == Interval(Locus("X", 1), Locus("X", 101), false, true))
    assert(Locus.parseInterval("X:START-101", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("X:START-Y:END", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, true))
    assert(Locus.parseInterval("X-Y", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, true))
    assert(Locus.parseInterval("1-22", rg) == Interval(Locus("1", 1), Locus("22", chr22Max), true, true))

    // test normalizing end points
    assert(Locus.parseInterval(s"(X:100-${ xMax + 1 })", rg) == Interval(Locus("X", 100), Locus("X", xMax), false, true))
    assert(Locus.parseInterval(s"(X:0-$xMax]", rg) == Interval(Locus("X", 1), Locus("X", xMax), true, true))
    TestUtils.interceptFatal("Start `X:0' is not within the range")(Locus.parseInterval("[X:0-5)", rg))
    TestUtils.interceptFatal(s"End `X:${ xMax + 1 }' is not within the range")(Locus.parseInterval(s"[X:1-${ xMax + 1 }]", rg))

    assert(Locus.parseInterval("[16:29500000-30200000)", rg) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[16:29.5M-30.2M)", rg) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[16:29500K-30200K)", rg) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[1:100K-2:200K)", rg) == Interval(Locus("1", 100000), Locus("2", 200000), true, false))

    assert(Locus.parseInterval("[1:1.111K-2000)", rg) == Interval(Locus("1", 1111), Locus("1", 2000), true, false))
    assert(Locus.parseInterval("[1:1.111111M-2000000)", rg) == Interval(Locus("1", 1111111), Locus("1", 2000000), true, false))

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("4::start-5:end", rg)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("4:start-", rg)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("1:1.1111K-2k", rg)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("1:1.1111111M-2M", rg)
    }

    val gr37 = ReferenceGenome.GRCh37
    val gr38 = ReferenceGenome.GRCh38

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

  @Test def testFilterIntervals() {
    val ds = hc.importVCF("src/test/resources/sample.vcf", nPartitions=Some(20))
    val intervals = Array(Interval(Row(Locus("20", 10019093)), Row(Locus("20", 10026348)), true, true),
      Interval(Row(Locus("20", 17705793)), Row(Locus("20", 17716416)), true, true))
    val iTree = IntervalTree(ds.rvd.typ.pkType.ordering, intervals)
    assert(FilterIntervals(ds, iTree, true).countRows() == 4)
  }
}
