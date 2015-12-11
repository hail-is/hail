package org.broadinstitute.hail.variant.vsm

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.io.LoadVCF
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import scala.collection.mutable
import scala.util.Random
import scala.language.postfixOps
import org.testng.annotations.Test

class VSMSuite extends SparkSuite {

  @Test def testSame() {
    val vds1 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vds2 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    assert(vds1.same(vds2))

    val mdata1 = VariantMetadata(Map("1" -> 10, "2" -> 10), IndexedSeq("S1", "S2", "S3"))
    val mdata2 = VariantMetadata(Map("1" -> 10, "2" -> 20), IndexedSeq("S1", "S2", "S3"))
    val mdata3 = VariantMetadata(Map("1" -> 10), IndexedSeq("S1", "S2"))

    assert(mdata1 != mdata2)
    assert(mdata1 != mdata3)
    assert(mdata2 != mdata3)

    val v1 = Variant("1", 1, "A", "T")
    val v2 = Variant("1", 2, "T", "G")
    val v3 = Variant("1", 2, "T", "A")

    val rdd1 = sc.parallelize(Seq(v1 ->
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0))),
      v2 ->
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50)))))

    // differ in variant
    val rdd2 = sc.parallelize(Seq(v1 ->
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0))),
      v3 ->
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50)))))

    // differ in genotype
    val rdd3 = sc.parallelize(Seq(v1 ->
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(1, (7, 8), 15, (100, 0, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0))),
      v2 ->
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50)))))

    // for mdata3
    val rdd4 = sc.parallelize(Seq(v1 ->
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100))),
      v2 -> Iterable(
        Genotype(0, (10, 0), 10, (0, 10, 100)),
        Genotype(0, (11, 0), 11, (0, 10, 100)))))

    // differ in number of variants
    val rdd5 = sc.parallelize(Seq(v1 ->
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)))))

    val vdss = Array(new VariantDataset(mdata1, rdd1),
      new VariantDataset(mdata1, rdd2),
      new VariantDataset(mdata1, rdd3),
      new VariantDataset(mdata2, rdd1),
      new VariantDataset(mdata2, rdd2),
      new VariantDataset(mdata2, rdd3),
      new VariantDataset(mdata3, rdd4),
      new VariantDataset(mdata3, rdd5))

    for (i <- vdss.indices;
      j <- vdss.indices) {
      if (i == j)
        assert(vdss(i) == vdss(j))
      else
        assert(vdss(i) != vdss(j))
    }
  }

  @Test def testFilterSamples() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vdsAsMap = vds.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
    val nSamples = vds.nSamples
    assert(nSamples == vds.nLocalSamples)

    // FIXME ScalaCheck
    for (n <- 0 until 20) {
      val keep = mutable.Set.empty[Int]

      // n == 0: none
      if (n == 1) {
        for (i <- 0 until nSamples)
          keep += i
      } else if (n > 1) {
        for (i <- 0 until nSamples) {
          if (Random.nextFloat() < 0.5)
            keep += i
        }
      }

      val localKeep = keep
      val filtered = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
        .filterSamples(s => localKeep(s))

      val filteredAsMap = filtered.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
      filteredAsMap.foreach { case (k, g) => simpleAssert(vdsAsMap(k) == g) }

      simpleAssert(filtered.nSamples == nSamples)
      simpleAssert(filtered.localSamples.toSet == keep)

      val sampleKeys = filtered.mapWithKeys((v, s, g) => s).distinct.collect()
      assert(sampleKeys.toSet == keep)
    }
  }
}
