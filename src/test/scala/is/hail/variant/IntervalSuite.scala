package is.hail.variant

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.{Gen, Prop}
import is.hail.driver._
import is.hail.expr.{TSet, TString}
import is.hail.io.annotators.IntervalListAnnotator
import is.hail.io.vcf.LoadVCF
import is.hail.utils._
import org.testng.annotations.Test

import scala.io.Source

class IntervalSuite extends SparkSuite {

  def genomicInterval(contig: String, start: Int, end: Int): Interval[Locus] =
    Interval(Locus(contig, start),
      Locus(contig, end))

  @Test def test() {
    val ilist = IntervalTree(Array(
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

    val ex1 = IntervalListAnnotator.read("src/test/resources/example1.interval_list", hadoopConf)

    val f = tmpDir.createTempFile("example", extension = ".interval_list")
    IntervalListAnnotator.write(ex1, f, hadoopConf)
    val ex1wr = IntervalListAnnotator.read(f, hadoopConf)
    assert(ex1wr == ex1)

    val ex2 = IntervalListAnnotator.read("src/test/resources/example2.interval_list", hadoopConf)
    assert(ex1 == ex2)
  }

  @Test def testAll() {
    val vds = VariantSampleMatrix(hc, VariantMetadata(Array.empty[String]),
      sc.parallelize(Seq((Variant("1", 100, "A", "T"), (Annotation.empty, Iterable.empty[Genotype])))).toOrderedRDD)

    val intervalFile = tmpDir.createTempFile("intervals")
    hadoopConf.writeTextFile(intervalFile) { out =>
      out.write("1\t50\t150\t+\tTHING1\n")
      out.write("1\t50\t150\t+\tTHING2\n")
      out.write("1\t50\t150\t+\tTHING3\n")
      out.write("1\t50\t150\t+\tTHING4\n")
      out.write("1\t50\t150\t+\tTHING5")
    }

    assert(vds.annotateVariantsIntervals(intervalFile, "va", all = true)
      .variantsAndAnnotations
      .collect()
      .head._2 == Set("THING1", "THING2", "THING3", "THING4", "THING5"))
  }

  @Test def testNew() {
    val a = Array(
      genomicInterval("1", 10, 20),
      genomicInterval("1", 30, 40),
      genomicInterval("2", 40, 50))

    val t = IntervalTree[Locus](a)

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

    val t2 = IntervalTree[Locus](a2)

    assert(t2.query(Locus("1", 30)) == Set(
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
      val intervals = it.toSet

      val setResults = intervals.filter(_.contains(locus))
      val treeResults = it.query(locus)

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
      .annotateVariantsIntervals("src/test/resources/annotinterall.interval_list", "va.annot", all = true)

    val (t, q) = vds.queryVA("va.annot")
    assert(t == TSet(TString))

    vds.rdd.foreach { case (v, (va, gs)) =>
      val a = q(va).get.asInstanceOf[Set[String]]

      if (v.start == 17348324)
        simpleAssert(a == Set("A", "B"))
      else if (v.start >= 17333902 && v.start <= 17370919)
        simpleAssert(a == Set("A"))
      else
        simpleAssert(a.isEmpty)
    }

    @Test def testAggregate() {
      val vds = hc.importVCF("src/test/resources/sample2.vcf", nPartitions = Some(2)).cache()
        .splitMulti()
      val iList = tmpDir.createTempFile("input", ".interval_list")
      val tmp1 = tmpDir.createTempFile("output", ".tsv")

      val startPos = 16050036 - 250000
      val endPos = 17421565 + 250000
      val intervalGen = for (start <- Gen.choose(startPos, endPos);
        end <- Gen.choose(start, endPos))
        yield Interval(Locus("22", start), Locus("22", end))
      val intervalsGen = Gen.buildableOfN[Array, Interval[Locus]](500, intervalGen)

      val p = Prop.forAll(intervalsGen) { intervals =>

        hadoopConf.writeTextFile(iList) { out =>
          intervals.foreach { i =>
            out.write(s"22\t${ i.start.position }\t${ i.end.position }\n")
          }
        }

        vds.aggregateIntervals(iList, "nSNP = variants.filter(v => v.altAllele.isSNP).count(), " +
          "nIndel = variants.filter(v => v.altAllele.isIndel).count(), " +
          "N = variants.count()", tmp1)

        val variants = vds
          .variants
          .collect()

        hadoopConf.readFile(tmp1) { in =>

          Source.fromInputStream(in)
            .getLines()
            .toArray
            .tail
            .forall { line =>
              val Array(contig, startStr, endStr, nSnpStr, nIndelStr, nStr) = line.split("\t")
              val start = startStr.toInt
              val end = endStr.toInt
              val nSnp = nSnpStr.toInt
              val nIndel = nIndelStr.toInt
              val n = nStr.toInt

              val interval = Interval(Locus(contig, start), Locus(contig, end))
              val inInterval = variants.filter(v => interval.contains(v.locus))

              (nSnp == inInterval.count(_.altAllele.isSNP)) &&
                (nIndel == inInterval.count(_.altAllele.isIndel)) &&
                (n == inInterval.length)
            }
        }
      }

      p.check()
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
      g <- Gen.buildableOfN[Array, Interval[Locus]](nIntervals, intervalGen)) yield g

    Prop.forAll(intervalsGen) { intervals =>
      hadoopConf.writeTextFile(iList) { out =>
        intervals.foreach { i =>
          out.write(s"22\t${ i.start.position }\t${ i.end.position }\n")
        }
      }

      val vdsKeep = vds.filterIntervals(iList, keep = true)
      val vdsRemove = vds.filterIntervals(iList, keep = false)

      val p1 = vdsKeep.same(vds.copy(rdd = vds.rdd.filter { case (v, _) =>
        intervals.exists(_.contains(v.locus))
      }.asOrderedRDD))
      val p2 = vdsRemove.same(vds.copy(rdd = vds.rdd.filter { case (v, _) =>
        intervals.forall(!_.contains(v.locus))
      }.asOrderedRDD))
      p1 && p1
    }.check()
  }
}