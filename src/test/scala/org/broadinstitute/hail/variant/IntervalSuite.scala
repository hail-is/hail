package org.broadinstitute.hail.variant

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr.{TSet, TString}
import org.broadinstitute.hail.io.annotators.IntervalListAnnotator
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.utils._
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
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample2.vcf"))
    s = AnnotateVariantsIntervals.run(s, Array(
      "-i", "src/test/resources/annotinterall.interval_list", "-a", "-r", "va.annot"))

    val (t, q) = s.vds.queryVA("va.annot")
    assert(t == TSet(TString))

    s.vds.rdd.foreach { case (v, (va, gs)) =>
      val a = q(va).get.asInstanceOf[Set[String]]

      if (v.start == 17348324)
        simpleAssert(a == Set("A", "B"))
      else if (v.start >= 17333902 && v.start <= 17370919)
        simpleAssert(a == Set("A"))
      else
        simpleAssert(a.isEmpty)
    }

    @Test def testAggregate() {
      val vds = LoadVCF(sc, "src/test/resources/sample2.vcf", nPartitions = Some(2)).cache()
      val state = SplitMulti.run(State(sc, sqlContext, vds))
      val iList = tmpDir.createTempFile("input", ".interval_list")
      val tmp1 = tmpDir.createTempFile("output", ".tsv")

      val startPos = 16050036 - 250000
      val endPos = 17421565 + 250000
      val intervalGen = for (start <- Gen.choose(startPos, endPos);
                             end <- Gen.choose(start, endPos))
        yield Interval(Locus("22", start), Locus("22", end))
      val intervalsGen = Gen.buildableOfN[Array[Interval[Locus]], Interval[Locus]](500, intervalGen)

      val p = Prop.forAll(intervalsGen) { intervals =>

        writeTextFile(iList, hadoopConf) { out =>
          intervals.foreach { i =>
            out.write(s"22\t${ i.start.position }\t${ i.end.position }\n")
          }
        }

        AggregateIntervals.run(state, Array(
          "-i", iList,
          "-o", tmp1,
          "-c", "nSNP = variants.count(v.altAllele.isSNP), nIndel = variants.count(v.altAllele.isIndel), N = variants.count(true)"))

        val variants = state
          .vds
          .variants
          .collect()

        readFile(tmp1, hadoopConf) { in =>

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

      p.check(count = 100)
    }
  }
}