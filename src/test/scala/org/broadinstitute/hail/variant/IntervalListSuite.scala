package org.broadinstitute.hail.variant

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.check.Prop
import org.broadinstitute.hail.driver.{AggregateIntervals, SplitMulti, State}
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.Utils._
import org.testng.annotations.Test

import scala.io.Source

class IntervalListSuite extends SparkSuite {

  @Test def test() {
    val ilist = GenomicIntervalSet(Set(
      GenomicInterval("1", 10, 20),
      GenomicInterval("1", 30, 40),
      GenomicInterval("2", 40, 50)))

    assert(!ilist.contains("1", 5))
    assert(ilist.contains("1", 10))
    assert(ilist.contains("1", 15))
    assert(ilist.contains("1", 20))
    assert(!ilist.contains("1", 25))
    assert(ilist.contains("1", 35))

    assert(!ilist.contains("2", 30))
    assert(ilist.contains("2", 45))

    assert(!ilist.contains("3", 0))

    val ex1 = GenomicIntervalSet.read("src/test/resources/example1.interval_list", hadoopConf)

    val f = tmpDir.createTempFile("example", extension = ".interval_list")
    ex1.write(f, hadoopConf)
    val ex1wr = GenomicIntervalSet.read(f, hadoopConf)
    assert(ex1wr == ex1)

    val ex2 = GenomicIntervalSet.read("src/test/resources/example2.interval_list", hadoopConf)
    assert(ex1 == ex2)
  }

  @Test def testNew() {
    val a = Set(
      GenomicInterval("1", 10, 20),
      GenomicInterval("1", 30, 40),
      GenomicInterval("2", 40, 50))

    val t = GenomicIntervalSet(a)

    assert(t.contains("1", 15))
    assert(t.contains("1", 30))
    assert(t.contains("1", 40))
    assert(t.contains("1", 20))
    assert(!t.contains("1", 25))
    assert(!t.contains("1", 9))
    assert(!t.contains("5", 20))

    // Test queries
    val a2 = Set(
      GenomicInterval("1", 10, 20),
      GenomicInterval("1", 30, 40),
      GenomicInterval("1", 30, 50),
      GenomicInterval("1", 29, 31),
      GenomicInterval("1", 29, 30),
      GenomicInterval("2", 41, 50),
      GenomicInterval("2", 43, 50),
      GenomicInterval("2", 45, 70),
      GenomicInterval("2", 42, 43),
      GenomicInterval("2", 1, 10))

    val t2 = GenomicIntervalSet(a2)

    assert(t2.query("1", 30).toSet == Set(
      GenomicInterval("1", 30, 40), GenomicInterval("1", 30, 50),
      GenomicInterval("1", 29, 31), GenomicInterval("1", 29, 30)))

  }

  @Test def random() {

    val g = for (t <- IntervalSet.gen;
                 queries <- IntervalSet.genQueries(t)) yield (t, queries)

    val p = Prop.forAll(g) { case (t, queries) =>
      val intervals = t.intervals
      queries.forall { i =>
        val inSet = intervals.exists(_.contains(i))
        val inTree = t.contains(i)
        if (inSet != inTree)
          println(
            s"""contains results not the same for position [ $i ]
                |  from set:  $inSet
                |  from tree: $inTree
             """.stripMargin)

        val setQueries = intervals.filter(_.contains(i))
        val treeQueries = t.query(i)
        if (treeQueries != setQueries)
          println(
            s"""query results not the same for query position [ $i ]
                |  from set:  $setQueries
                |  from tree: $treeQueries
             """.stripMargin)
        (inSet == inTree) && (treeQueries == setQueries)
      }
    }
    p.check()
  }

  @Test def testAggregate() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds))
    val tmp1 = tmpDir.createTempFile("output", ".tsv")
    AggregateIntervals.run(state, Array(
      "-i", "src/test/resources/exampleAnnotation1.interval_list",
      "-o", tmp1,
      "-c", "nSNP = variants.count(v.altAllele.isSNP), nIndel = variants.count(v.altAllele.isIndel), N = variants.count(true)"))

    val variants = state
      .vds
      .variants
      .collect()

    readFile(tmp1, hadoopConf) { in =>

      val Array(header, l1, l2) = Source.fromInputStream(in)
        .getLines()
        .toArray

      val Array(_, _, _, nSnp1, nIndel1, n1) = l1.split("\t")
      val Array(_, _, _, nSnp2, nIndel2, n2) = l2.split("\t")

      assert(nSnp1.toInt == variants.count(v => v.start >= 1 && v.start <= 14000000 && v.altAllele.isSNP))
      assert(nIndel1.toInt == variants.count(v => v.start >= 1 && v.start <= 14000000 && v.altAllele.isIndel))
      assert(n1.toInt == variants.count(v => v.start >= 1 && v.start <= 14000000))

      assert(nSnp2.toInt == variants.count(v => v.start >= 17000000 && v.start <= 18000000 && v.altAllele.isSNP))
      assert(nIndel2.toInt == variants.count(v => v.start >= 17000000 && v.start <= 18000000 && v.altAllele.isIndel))
      assert(n2.toInt == variants.count(v => v.start >= 17000000 && v.start <= 18000000))
    }
  }
}
