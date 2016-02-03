package org.broadinstitute.hail.variant.vsm

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import scala.collection.mutable
import scala.util.Random
import scala.language.postfixOps
import org.broadinstitute.hail.methods.LoadVCF
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Arbitrary._
import org.broadinstitute.hail.annotations._

class VSMSuite extends SparkSuite {

  @Test def testSame() {
    val vds1 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vds2 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    assert(vds1.same(vds2))

    val mdata1 = VariantMetadata(Array("S1", "S2", "S3"))
    val mdata2 = VariantMetadata(Array("S1", "S2"))
    val mdata3 = new VariantMetadata(IndexedSeq.empty[(String, String)], Array("S1", "S2"),
      Annotations.emptyIndexedSeq(2).map(_ + ("1", "5")), Annotations.empty(),
      Annotations.empty())
    val mdata4 = new VariantMetadata(IndexedSeq.empty[(String, String)], Array("S1", "S2"),
      Annotations.emptyIndexedSeq(2), Annotations.empty(), Annotations.empty()
        + ("dummy", Map.empty[String, AnnotationSignature]))

    assert(mdata1 != mdata2)

    val v1 = Variant("1", 1, "A", "T")
    val v2 = Variant("1", 2, "T", "G")
    val v3 = Variant("1", 2, "T", "A")

    val va1 = Annotations(Map("info" -> Map("v1thing" -> "yes"), "v1otherThing" -> "yes"))
    val va2 = Annotations(Map("info" -> Map("v1thing" -> "yes"), "v2otherThing" -> "yes"))
    val va3 = Annotations(Map("info" -> Map("v1thing" -> "yes"), "v1otherThing" -> "no"))

    val rdd1 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2))),
      (v2, va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))

    // differ in variant
    val rdd2 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2))),
      (v3, va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))

    // differ in genotype
    val rdd3 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(),
        Genotype(1),
        Genotype(2))),
      (v2, va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))

    // for mdata2
    val rdd4 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(),
        Genotype(0))),
      (v2, va2, Iterable(
        Genotype(0),
        Genotype(0)))))

    // differ in number of variants
    val rdd5 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(),
        Genotype(0)))))

    // differ in annotations
    val rdd6 = sc.parallelize(Seq((v1, va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2))),
      (v2, va3,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))

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

  @Test def testReadWrite() {
    val p = forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)) { (vsm: VariantSampleMatrix[Genotype]) =>
      hadoopDelete("/tmp/foo.vds", sc.hadoopConfiguration, recursive = true)
      vsm.write(sqlContext, "/tmp/foo.vds")
      val vsm2 = VariantSampleMatrix.read(sqlContext, "/tmp/foo.vds")
      vsm2.same(vsm)
    }

    p.check
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
