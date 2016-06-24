package org.broadinstitute.hail.variant

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.driver.{AggregateIntervals, SplitMulti, State}
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.utils._
import org.testng.annotations.Test

import scala.io.Source

class IntervalSuite extends SparkSuite {

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
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf").cache()
    val state = SplitMulti.run(State(sc, sqlContext, vds))
    val iList = tmpDir.createTempFile("input", ".interval_list")
    val tmp1 = tmpDir.createTempFile("output", ".tsv")

    val startPos = 16050036 - 250000
    val endPos = 17421565 + 250000
    val g = Gen.parameterized { p =>
      val rng = p.rng
      Gen.const((0 until 500).map { _ =>
        val start = rng.nextInt(startPos, endPos)
        val end = rng.nextInt(start, endPos)
        Interval(start, end)
      })
    }
    val p = Prop.forAll(g) { intervals =>
      writeTextFile(iList, hadoopConf) { out =>
        intervals.foreach { i =>
          out.write(s"22\t${i.start}\t${i.end}\n")
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
            val Array(_, startStr, endStr, nSnpStr, nIndelStr, nStr) = line.split("\t")
            val start = startStr.toInt
            val end = endStr.toInt
            val nSnp = nSnpStr.toInt
            val nIndel = nIndelStr.toInt
            val n = nStr.toInt

            (nSnp == variants.count(v => v.start >= start && v.start <= end && v.altAllele.isSNP)) &&
              (nIndel == variants.count(v => v.start >= start && v.start <= end && v.altAllele.isIndel)) &&
              (n == variants.count(v => v.start >= start && v.start <= end))
          }
      }
    }

    p.check(count = 10)
  }
}
