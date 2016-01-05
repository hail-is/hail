package org.broadinstitute.hail.variant.vsm

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import scala.collection.mutable
import scala.util.Random
import scala.language.postfixOps
import org.broadinstitute.hail.methods.LoadVCF
import org.testng.annotations.Test
import org.broadinstitute.hail.annotations._

class VSMSuite extends SparkSuite {

  @Test def testSame() {
    val vds1 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vds2 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    assert(vds1.same(vds2))

    val mdata1 = VariantMetadata(Array("S1", "S2", "S3"))
    val mdata2 = VariantMetadata(Array("S1", "S2"))
    val mdata3 = new VariantMetadata(Seq.empty[(String, String)], Array("S1", "S2"), None,
      Annotations.emptyOfArrayString(2).map(_.addVal("1", "5")), Annotations.emptyOfSignature(),
      Annotations.emptyOfSignature())
    val mdata4 = new VariantMetadata(Seq.empty[(String, String)], Array("S1", "S2"), None,
      Annotations.emptyOfArrayString(2), Annotations.emptyOfSignature(), Annotations.emptyOfSignature()
        .addMap("dummy", Map.empty[String, AnnotationSignature]))

    assert(mdata1 != mdata2)

    val v1 = Variant("1", 1, "A", "T")
    val v2 = Variant("1", 2, "T", "G")
    val v3 = Variant("1", 2, "T", "A")

    val va1 = Annotations(Map("info" -> Map("v1thing" -> "yes")), Map("v1otherThing" -> "yes"))
    val va2 = Annotations(Map("info" -> Map("v2thing" -> "yes")), Map("v2otherThing" -> "yes"))
    val va3 = Annotations(Map("info" -> Map("v1thing" -> "no")), Map("v1otherThing" -> "no"))

    val rdd1 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0)))),
      (v2, va2,
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50))))))

    // differ in variant
    val rdd2 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0)))),
      (v3, va2,
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50))))))

    // differ in genotype
    val rdd3 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(1, (7, 8), 15, (100, 0, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0)))),
      (v2, va2,
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50))))))

    // for mdata2
    val rdd4 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)))),
      (v2, va2, Iterable(
        Genotype(0, (10, 0), 10, (0, 10, 100)),
        Genotype(0, (11, 0), 11, (0, 10, 100))))))

    // differ in number of variants
    val rdd5 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100))))))

    // differ in annotations
    val rdd6 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(-1, (0, 2), 2, null),
        Genotype(0, (11, 1), 12, (0, 10, 100)),
        Genotype(2, (0, 13), 13, (100, 10, 0)))),
      (v2, va3,
        Iterable(Genotype(0, (10, 0), 10, (0, 10, 100)),
          Genotype(0, (11, 0), 11, (0, 10, 100)),
          Genotype(1, (6, 6), 12, (50, 0, 50))))))

    val vdss = Array(new VariantDataset(mdata1, rdd1),
      new VariantDataset(mdata1, rdd2),
      new VariantDataset(mdata1, rdd3),
      new VariantDataset(mdata2, rdd1),
      new VariantDataset(mdata2, rdd2),
      new VariantDataset(mdata2, rdd3),
      new VariantDataset(mdata2, rdd4),
      new VariantDataset(mdata2, rdd5),
      new VariantDataset(mdata3, rdd1),
      new VariantDataset(mdata3, rdd2),
      new VariantDataset(mdata4, rdd1),
      new VariantDataset(mdata4, rdd2),
      new VariantDataset(mdata1, rdd6))

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
      // FIXME
      .mapAnnotations((v, va) => Annotations.emptyOfData())
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
      val filtered = vds.filterSamples((s, sa) => localKeep(s))

      val filteredAsMap = filtered.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
      filteredAsMap.foreach { case (k, g) => simpleAssert(vdsAsMap(k) == g) }

      simpleAssert(filtered.nSamples == nSamples)
      simpleAssert(filtered.localSamples.toSet == keep)

      val sampleKeys = filtered.mapWithKeys((v, s, g) => s).distinct.collect()
      assert(sampleKeys.toSet == keep)

      val filteredOut = "/tmp/test_filtered.vds"
      hadoopDelete(filteredOut, sc.hadoopConfiguration, true)
      filtered.write(sqlContext, filteredOut, compress = true)

      val filtered2 = VariantSampleMatrix.read(sqlContext, filteredOut)
      assert(filtered2.same(filtered))
    }
  }
}
