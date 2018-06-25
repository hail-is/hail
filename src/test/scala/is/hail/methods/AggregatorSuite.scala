package is.hail.methods

import is.hail.check.{Gen, Prop}
import is.hail.expr._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.testUtils._
import is.hail.utils._
import is.hail.variant.{MatrixTable, VSMSubgen}
import is.hail.{SparkSuite, TestUtils}
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class AggregatorSuite extends SparkSuite {

  @Test def testSum() {
    val p = Prop.forAll(MatrixTable.gen(hc, VSMSubgen.plinkSafeBiallelic)) { vds =>
      var vds2 = VariantQC(vds, "qc")
      vds2 = vds2
        .annotateRowsExpr("oneHotAC" -> "AGG.map(g => g.GT.oneHotAlleles(va.alleles.length()).map(x => x.toInt64())).sum().map(x => x.toInt32())")
        .annotateRowsExpr("same" -> ("(AGG.filter(g => isDefined(g.GT)).count() == 0L) || " +
          "(va.oneHotAC[0] == va.qc.n_called * 2  - va.qc.AC) && (va.oneHotAC[1] == va.qc.n_het + 2 * va.qc.n_hom_var)"))
      vds2.rowsTable().forall("row.same")
    }
    p.check()
  }

  @Test def testMaxMin() {
    val rdd = sc.parallelize(Seq(
      Row("a", 0, null, 1, -1, null, null, 1, null, 0l, 0f, 0d),
      Row("a", -1, -1, null, -2, null, 1, null, null, -1l, -1f, -1d),
      Row("a", 1, -2, 2, null, -1, null, null, null, 1l, 1f, 1d)), numSlices = 2)

    val signature = TStruct((("group" -> TString()) +: (0 until 8).map(i => s"s$i" -> TInt32()))
      ++ IndexedSeq("s8" -> TInt64(), "s9" -> TFloat32(), "s10" -> TFloat64()): _*)

    val ktMax = Table(hc, rdd, signature, key = Some(IndexedSeq("group")))
      .aggregateByKey(
        (0 until 11).map(i => s"s$i : AGG.map(r => r.s$i).max()").mkString("{ ",", ", " }"),
        (0 until 11).map(i => s"s$i = AGG.map(r => r.s$i).max()").mkString(", "))

    assert(ktMax.collect() sameElements Array(Row("a", 1, -1, 2, -1, -1, 1, 1, null, 1l, 1f, 1d)))

    val ktMin = Table(hc, rdd, signature, key = Some(IndexedSeq("group")))
      .aggregateByKey(
        (0 until 11).map(i => s"s$i : AGG.map(r => r.s$i).min()").mkString("{ ", ", ", " }"),
        (0 until 11).map(i => s"s$i = AGG.map(r => r.s$i).min()").mkString(","))

    assert(ktMin.collect() sameElements Array(Row("a", -1, -2, 1, -2, -1, 1, 1, null, -1l, -1f, -1d)))
  }

  @Test def testProduct() {
    val rdd = sc.parallelize(Seq(
      Row("a", 0l, null, 1l, 1l, null, null, 10l, null, 0l, 2d, 0d),
      Row("a", -1l, -1l, null, 2l, null, 1l, 4l, null, -1l, -1d, -1d),
      Row("a", 1l, -2l, 2l, 3l, -1l, -3l, 2l, null, 1l, 2d, 1d)), numSlices = 2)

    val signature = TStruct((("group" -> TString()) +: (0 until 8).map(i => s"s$i" -> TInt64()))
      ++ IndexedSeq("s8" -> TInt64(), "s9" -> TFloat64(), "s10" -> TFloat64()): _*)
    
    val ktProduct = Table(hc, rdd, signature, key = Some(IndexedSeq("group")))
      .aggregateByKey(
        ((0 until 11).map(i => s"s$i : AGG.map(r => r.s$i).product()")
          :+ "empty : AGG.map(r => r.s10).filter(x => false).product()").mkString("{ ", ", ", " }"),
        ((0 until 11).map(i => s"s$i = AGG.map(r => r.s$i).product()")
          :+ "empty = AGG.map(r => r.s10).filter(x => false).product()").mkString(", "))

    assert(ktProduct.collect() sameElements Array(Row("a", 0l, 2l, 2l, 6l, -1l, -3l, 80l, 1l, 0l, -4d, 0d, 1d)))
  }

  @Test def testHist() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf").cache()

    assert(vds.annotateRowsExpr(
        "hist" -> "AGG.map(g => g.GQ.toFloat64).hist(0.0, 100.0, 20)",
        "bin0" -> "AGG.filter(g => g.GQ < 5).count()",
        "bin1" -> "AGG.filter(g => g.GQ >= 5 && g.GQ < 10).count()",
        "last" -> "AGG.filter(g => g.GQ >= 95).count()")
      .rowsTable()
      .forall(
        """
        row.hist.bin_freq[0] == row.bin0 &&
        row.hist.bin_freq[1] == row.bin1 &&
        row.hist.bin_freq[-1] == row.last"""))

    assert(vds
      .annotateRowsExpr(
        "hist" -> "AGG.map(g => g.GQ.toFloat64).hist(22.0, 80.0, 5)",
        "nLess" -> "AGG.filter(g => g.GQ < 22).count()",
        "nGreater" -> "AGG.filter(g => g.GQ > 80).count()")
      .rowsTable()
      .forall(
        """
        row.hist.n_smaller == row.nLess &&
        row.hist.n_larger == row.nGreater"""))
  }

  @Test def testCallStats() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf").cache()
      .annotateRowsExpr(
        "callStats" -> "AGG.map(g => g.GT).callStats(g => va.alleles.length)",
          "AC" -> "AGG.map(g => g.GT.oneHotAlleles(va.alleles.length()).map(x => x.toInt64())).sum().map(x => x.toInt32())",
          "AN" -> "AGG.filter(g => isDefined(g.GT)).count() * 2L")
      .annotateRowsExpr("AF" -> "va.AC.map(x => x.toFloat64) / va.AN.toFloat64()")
    val (_, csAC) = vds.queryVA("va.callStats.AC")
    val (_, csAF) = vds.queryVA("va.callStats.AF")
    val (_, csAN) = vds.queryVA("va.callStats.AN")
    val (_, ac) = vds.queryVA("va.AC")
    val (_, af) = vds.queryVA("va.AF")
    val (_, an) = vds.queryVA("va.AN")

    vds.variantsAndAnnotations
      .collect()
      .foreach { case (_, va) =>
        assert(csAC(va) == ac(va), s"AC was different")
        assert(csAN(va) == an(va), s"AN was different")
        assert(csAF(va) == af(va), s"AF was different")
      }
  }

  @Test def testCounter() {
    Prop.forAll(MatrixTable.gen(hc, VSMSubgen.plinkSafeBiallelic)) { vds =>
      val (r, t) = vds.aggregateRows("AGG.map(_ => va.locus.contig).counter()")
      val counterMap = r.asInstanceOf[Map[String, Long]]
      val aggMap = vds.variants.map(_.asInstanceOf[Variant].contig).countByValue()
      aggMap == counterMap
    }.check()
  }

  @Test def testTake() {
    val vds = hc.importVCF("src/test/resources/aggTake.vcf")
      .annotateRowsExpr("take" -> "AGG.map(g => g.DP).take(3)")
      .annotateRowsExpr("takeBy" -> "AGG.map(g => g.DP).takeBy(dp => g.GQ, 3)")

    val (_, qTake) = vds.queryVA("va.take")
    val (_, qTakeBy) = vds.queryVA("va.takeBy")

    val va = vds.variantsAndAnnotations.map(_._2).collect().head
    assert(qTake(va) == IndexedSeq[Any](11, null, 20))
    assert(qTakeBy(va) == IndexedSeq[Any](5, 20, 1))
  }

  private def isLensedPrefix[T, K](lens: T => K)(prefix: Seq[T], full: Seq[T]): Boolean = {
    prefix.zip(full).forall { case (x, y) => lens(x) == lens(y) }
  }

  private def prefixModuloDisordering[T, K](sortBy: T => K)(prefix: Seq[T], full: Seq[T]): Boolean = {
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
      val wholeClassesFull = fullEquivClasses.filterKeys(k => wholeClassesPrefix.keySet.contains(k))

      val wholeClassesSame = wholeClassesFull == wholeClassesPrefix

      val lastKeySubset = prefixEquivClasses(lastKey).subsetOf(fullEquivClasses(lastKey))

      if (sameOrdering) {
        if (wholeClassesSame) {
          if (lastKeySubset) {
            true
          } else {
            println(s"The values at the last key in the prefix, $lastKey, were not a subset of those in the full list: ${ prefixEquivClasses(lastKey) } ${ fullEquivClasses(lastKey) }")
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

    Prop.forAll(MatrixTable.gen(hc, VSMSubgen.realistic)) { (vds: MatrixTable) =>
      val (a, _) = vds.aggregateEntries("AGG.collect().sortBy(g => g.GQ).map(g => [g.DP, g.GQ])")
      val (b, _) = vds.aggregateEntries("AGG.map(g => [g.DP, g.GQ]).takeBy(x => x[1], 10)")

      val sortby = a.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Integer]]]
      val takeby = b.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Integer]]]

      if (!prefixModuloDisordering((x: Seq[java.lang.Integer]) => x(1))(takeby, sortby)) {
        println(s"The first sequence is not a prefix, up to irrelevant disorderings, of the second sequence\n$takeby\n$sortby")
        false
      } else {
        true
      }
    }.check()
  }

  @Test def takeByAndSortByAgreeUsingLatentEnvironment() {
    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)

    Prop.forAll(MatrixTable.gen(hc, VSMSubgen.realistic)) { (vds: MatrixTable) =>
      vds.typecheck()

      val (a, _) = vds.aggregateEntries("AGG.collect().sortBy(g => g.GQ).map(g => [g.DP, g.GQ])")
      val (b, _) = vds.aggregateEntries("AGG.map(g => [g.DP, g.GQ]).takeBy(x => g.GQ, 10)")

      val sortby = a.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Integer]]]
      val takeby = b.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Integer]]]

      if (!prefixModuloDisordering((x: Seq[java.lang.Integer]) => x(1))(takeby, sortby)) {
        println(s"The first sequence is not a prefix, up to irrelevant disorderings, of the second sequence\n$takeby\n$sortby")
        false
      } else {
        true
      }
    }.check()
  }

  private val na = null

  private def doubleSeqEq(xs: Seq[java.lang.Double], ys: Seq[java.lang.Double]) =
    xs.zip(ys).forall {
      case (null, null) => true
      case (x, null) => false
      case (null, y) => false
      case (x, y) if x.isNaN && y.isNaN => true
      case (x, y) if x.isNaN => false
      case (x, y) if y.isNaN => false
      case (x, y) => x == y
    }

  @Test def testCollectAsSet() {
    val kt = Table.range(hc, 100, nPartitions = Some(10))

    assert(kt.aggregate("AGG.map(r => r.idx).collectAsSet()")._1 == (0 until 100).toSet)
    assert(kt.union(kt, kt).aggregate("AGG.map(r => r.idx).collectAsSet()")._1 == (0 until 100).toSet)
  }

  @Test def testArraySumInt64() {
    val kt = Table.range(hc, 100, nPartitions = Some(10))

    assert(kt.select("{foo : [row.idx.toInt64()]}", None, None)
      .aggregate("AGG.map(r => r.foo).sum()")._1
      == Seq((0 until 100).sum))
  }

  @Test def testArraySumFloat64() {
    val kt = Table.range(hc, 100, nPartitions = Some(10))

    assert(kt.select("{foo : [row.idx.toFloat64()/2.0]}", None, None)
      .aggregate("AGG.map(r => r.foo).sum()")._1
      == Seq((0 until 100).map(_ / 2.0).sum))
  }
}
