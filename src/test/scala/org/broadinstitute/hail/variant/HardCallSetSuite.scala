package org.broadinstitute.hail.variant

import breeze.linalg.DenseVector
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

import scala.util.Random

class HardCallSetSuite extends SparkSuite {
  @Test def callStreamTest() {

    def randGtArray(n: Int, pNonRef: Double): Array[Int] = {
      val a = Array.ofDim[Int](n)
      for (i <- 0 until n)
        a(i) =
          if (Random.nextDouble < pNonRef)
            Random.nextInt(3) + 1
          else 0
      a
    }

    val gtsList = List(
      Array(0),
      Array(0,1),
      Array(0,1,2),
      Array(0,1,2,3),
      Array(0,1,2,3,0),
      Array(0,1,2,3,1),
      Array(0,1,2,3,1,2),
      Array(0,1,2,3,0,0,0,0,0,0,0,0,0,0,1,2,3),
      randGtArray(100, .1),
      randGtArray(101, .1),
      randGtArray(102, .1),
      randGtArray(103, .1),
      randGtArray(1000, .01),
      randGtArray(1001, .01),
      randGtArray(1002, .01),
      randGtArray(1003, .01),
      randGtArray(10000, .001),
      randGtArray(10001, .001),
      randGtArray(10002, .001),
      randGtArray(10003, .001),
      randGtArray(100000, .0001),
      randGtArray(100001, .0001),
      randGtArray(100002, .0001),
      randGtArray(100003, .0001),
      randGtArray(1000001, .00001),
      randGtArray(1000002, .00001),
      randGtArray(1000003, .00001),
      randGtArray(1000004, .00001)
    )

    import CallStream._

    for (gts <- gtsList) {
      val n = gts.size
      val nNonRef = gts.count(_ != 0)

      val dcs = denseCallStreamFromGtStream(gts, nNonRef)
      val scs = sparseCallStreamFromGtStream(gts, nNonRef)

      val y = DenseVector.fill[Double](n)(Random.nextInt(2))

      val ds = dcs.hardStats(n, n, Array.fill(n)(true), (0 until n).toArray)
      val ss = scs.hardStats(n, n, Array.fill(n)(true), (0 until n).toArray)

      assert(ds == ss)

      //print(scs)
      //println(ss)
      //println()
    }
  }


  @Test def hcsTest() {
    val vds = LoadVCF(sc, "src/test/resources/regressionLinear.vcf")

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (2, ., 2, ., 0, 0)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val variantFilter = Set(v1, v2, v6, v7, v8, v9, v10)
    val sampleFilter = Set("A","B","C","D","E","F")

    val filtVds = vds
      .filterVariants((v, va, gs) => variantFilter(v))
      .filterSamples((s, sa) => sampleFilter(s))

    val hcs = HardCallSet(sqlContext, filtVds, sparseCutoff = .5)

    assert(hcs.nVariants == 7)
    assert(hcs.nSparseVariants == 2)
    assert(hcs.nDenseVariants == 5)
    assert(hcs.nSamples == 6)
    assert(hcs.sampleIds == filtVds.sampleIds)
    assert(hcs.sampleIds == filtVds.sampleIds)

    assert(hcs.df.groupBy("contig", "block").count().first().getLong(2) == 7)

    assert(hcs.capNVariantsPerBlock(3).df.groupBy("contig", "block").count().first().getLong(2) == 3)

    // hcs.writeNVariantsPerBlock("tmp/nVariantsPerBlock.tsv")
    // hcs.capNVariantsPerBlock(2).writeNVariantsPerBlock("tmp/nVariantsPerBlock.tsv")

    // FIXME: fails due to "file already exists" error on second run
    /*
    hcs.write(sqlContext, "/tmp/hardCallSet.hcs")
    val hcs2 = HardCallSet.read(sqlContext, "/tmp/hardCallSet.hcs")

    assert(hcs.sampleIds == hcs2.sampleIds)
    assert(hcs.sparseCutoff == hcs2.sparseCutoff)
    assert(hcs.blockWidth == hcs2.blockWidth)

    val hcsMap: Variant => CallStream = hcs.rdd.collect().toMap
    val hcs2Map: Variant => CallStream = hcs2.rdd.collect().toMap

    for (v <- variantFilter) {
      assert(hcsMap(v).a sameElements hcs2Map(v).a)
      assert(hcsMap(v).nNonRefForAllSamples == hcs2Map(v).nNonRefForAllSamples)
      assert(hcsMap(v).isSparse == hcs2Map(v).isSparse)
    }
    */

  }

}
