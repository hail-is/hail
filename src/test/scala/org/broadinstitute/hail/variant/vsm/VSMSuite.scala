package org.broadinstitute.hail.variant.vsm

import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.{Gen, Parameters}
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.sparkextras.OrderedRDD
import org.testng.annotations.Test

import scala.collection.mutable
import scala.language.postfixOps
import scala.util.Random

object VSMSuite {
  def checkOrderedRDD[T, K, V](rdd: OrderedRDD[T, K, V])(implicit kOrd: Ordering[K]): Boolean = {
    import Ordering.Implicits._

    case class PartInfo(min: K, max: K, isSorted: Boolean)

    val partInfo = rdd.mapPartitionsWithIndex { case (i, it) =>
      if (it.hasNext) {
        val s = it.map(_._1).toSeq

        Iterator((i, PartInfo(s.head, s.last, s.isSorted)))
      } else
        Iterator()
    }.collect()
      .sortBy(_._1)
      .map(_._2)

    partInfo.forall(_.isSorted) &&
      (partInfo.isEmpty ||
        partInfo.zip(partInfo.tail).forall { case (pi, pinext) =>
          pi.max < pinext.min
        })
  }
}

class VSMSuite extends SparkSuite {

  @Test def testSame() {
    val vds1 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vds2 = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    assert(vds1.same(vds2))

    val mdata1 = VariantMetadata(Array("S1", "S2", "S3"))
    val mdata2 = VariantMetadata(Array("S1", "S2"))
    val mdata3 = new VariantMetadata(
      Array("S1", "S2"),
      Annotation.emptyIndexedSeq(2),
      Annotation.empty,
      TStruct(
        "inner" -> TStruct(
          "thing1" -> TString),
        "thing2" -> TString),
      TStruct.empty,
      TStruct.empty)
    val mdata4 = new VariantMetadata(
      Array("S1", "S2"),
      Annotation.emptyIndexedSeq(2),
      Annotation.empty,
      TStruct(
        "inner" -> TStruct(
          "thing1" -> TString),
        "thing2" -> TString,
        "dummy" -> TString),
      TStruct.empty,
      TStruct.empty)

    assert(mdata1 != mdata2)
    assert(mdata1 != mdata3)
    assert(mdata2 != mdata3)
    assert(mdata1 != mdata4)
    assert(mdata2 != mdata4)
    assert(mdata3 != mdata4)

    val v1 = Variant("1", 1, "A", "T")
    val v2 = Variant("1", 2, "T", "G")
    val v3 = Variant("1", 2, "T", "A")

    val r1 = Annotation(Annotation("yes"), "yes")
    val r2 = Annotation(Annotation("yes"), "no")
    val r3 = Annotation(Annotation("no"), "yes")


    val va1 = r1
    val va2 = r2
    val va3 = r3

    val rdd1 = sc.parallelize(Seq((v1, (va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v2, (va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))).toOrderedRDD

    // differ in variant
    val rdd2 = sc.parallelize(Seq((v1, (va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v3, (va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))).toOrderedRDD

    // differ in genotype
    val rdd3 = sc.parallelize(Seq((v1, (va1,
      Iterable(Genotype(),
        Genotype(1),
        Genotype(2)))),
      (v2, (va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))).toOrderedRDD

    // for mdata2
    val rdd4 = sc.parallelize(Seq((v1, (va1,
      Iterable(Genotype(),
        Genotype(0)))),
      (v2, (va2, Iterable(
        Genotype(0),
        Genotype(0)))))).toOrderedRDD

    // differ in number of variants
    val rdd5 = sc.parallelize(Seq((v1, (va1,
      Iterable(Genotype(),
        Genotype(0)))))).toOrderedRDD

    // differ in annotations
    val rdd6 = sc.parallelize(Seq((v1, (va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v2, (va3,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1)))))).toOrderedRDD

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
    val p = forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random)) { vds =>
      val f = tmpDir.createTempFile(extension = ".vds")
      vds.write(sqlContext, f)
      val vds2 = VariantSampleMatrix.read(sqlContext, f)
      vds2.same(vds)
    }

    p.check()
  }

  @Test def testReadWriteUnpartitioned() {
    val p = forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random)) { vds =>
      val f = tmpDir.createTempFile(extension = ".vds")
      vds.write(sqlContext, f)
      hadoopConf.delete(f + "/partitioner", recursive = false)
      val vds2 = VariantSampleMatrix.read(sqlContext, f)
      vds2.same(vds)
    }

    p.check()
  }

  @Test(enabled = false) def testKuduReadWrite() {

    val vcf = "src/test/resources/multipleChromosomes.vcf"
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array(vcf))
    s = SplitMulti.run(s) // Kudu doesn't support multi-allelic yet

    val table = "variants_test"
    val master = "quickstart.cloudera"
    hadoopConf.delete("/tmp/foo.vds", recursive = true)

    s = WriteKudu.run(s, Array("-o", "/tmp/foo.vds", "-t", table, "-m", master,
      "--vcf-seq-dict", vcf, "--rows-per-partition", "300000000", "--drop"))

    val s2 = ReadKudu.run(s, Array("-i", "/tmp/foo.vds", "-t", table, "-m", master))

    assert(s.vds.same(s2.vds))
  }

  @Test def testFilterSamples() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vdsAsMap = vds.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
    val nSamples = vds.nSamples

    // FIXME ScalaCheck

    val samples = vds.sampleIds
    for (n <- 0 until 20) {
      val keep = mutable.Set.empty[String]

      // n == 0: none
      if (n == 1) {
        for (i <- 0 until nSamples)
          keep += samples(i)
      } else if (n > 1) {
        for (i <- 0 until nSamples) {
          if (Random.nextFloat() < 0.5)
            keep += samples(i)
        }
      }

      val localKeep = keep
      val filtered = vds.filterSamples((s, sa) => localKeep(s))

      val filteredAsMap = filtered.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
      filteredAsMap.foreach { case (k, g) => assert(vdsAsMap(k) == g) }

      assert(filtered.nSamples == keep.size)
      assert(filtered.sampleIds.toSet == keep)

      val sampleKeys = filtered.mapWithKeys((v, s, g) => s).distinct.collect()
      assert(sampleKeys.toSet == keep)

      val filteredOut = tmpDir.createTempFile("filtered", extension = ".vds")
      filtered.write(sqlContext, filteredOut, compress = true)

      val filtered2 = VariantSampleMatrix.read(sqlContext, filteredOut)
      assert(filtered2.same(filtered))
    }
  }

  @Test def testSkipGenotypes() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/sample2.vcf"))

    val f = tmpDir.createTempFile("sample", extension = ".vds")
    s = Write.run(s, Array("-o", f))

    s = Read.run(s, Array("--skip-genotypes", "-i", f))
    s = FilterVariantsExpr.run(s, Array("--keep", "-c", "va.info.AF[0] < 0.01"))

    assert(s.vds.nVariants == 234)
  }

  @Test def testSkipDropSame() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/sample2.vcf"))

    val f = tmpDir.createTempFile("sample", extension = ".vds")
    s = Write.run(s, Array("-o", f))

    s = Read.run(s, Array("--skip-genotypes", "-i", f))

    var s2 = Read.run(s, Array("-i", f))
    s2 = FilterSamplesAll.run(s)

    assert(s.vds.same(s2.vds))
  }

  @Test(enabled = false) def testVSMGenIsLinearSpaceInSizeParameter() {
    val minimumRSquareValue = 0.7
    def vsmOfSize(size: Int): VariantSampleMatrix[Genotype] = {
      val parameters = Parameters.default.copy(size = size, count = 1)
      VariantSampleMatrix.gen(sc, VSMSubgen.random).apply(parameters)
    }
    def spaceStatsOf[T](factory: () => T): SummaryStatistics = {
      val sampleSize = 50
      val memories = for (_ <- 0 until sampleSize) yield space(factory())._2

      val stats = new SummaryStatistics
      memories.foreach(x => stats.addValue(x.toDouble))
      stats
    }

    val sizes = 2500 to 20000 by 2500

    val statsBySize = sizes.map(size => (size, spaceStatsOf(() => vsmOfSize(size))))

    println("xs = " + sizes)
    println("mins = " + statsBySize.map { case (_, stats) => stats.getMin })
    println("maxs = " + statsBySize.map { case (_, stats) => stats.getMax })
    println("means = " + statsBySize.map { case (_, stats) => stats.getMean })

    val sr = new SimpleRegression
    statsBySize.foreach { case (size, stats) => sr.addData(size, stats.getMean) }

    println("R² = " + sr.getRSquare)

    assert(sr.getRSquare >= minimumRSquareValue,
      "The VSM generator seems non-linear because the magnitude of the R coefficient is less than 0.9")
  }

  @Test def testCoalesce() {
    val g = for (
      vsm <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
      k <- Gen.choose(1, math.max(1, vsm.nPartitions)))
      yield (vsm, k)

    forAll(g) { case (vsm, k) =>
      val coalesced = vsm.coalesce(k)
      VSMSuite.checkOrderedRDD(coalesced.rdd) && vsm.same(coalesced)
    }.check()
  }

  @Test def testUnionRead() {
    val g = for (vds <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
      variants <- Gen.const(vds.variants.collect());
      groups <- Gen.buildableOfN[Array, Int](variants.length, Gen.choose(1, 3)).map(groups => variants.zip(groups).toMap)
    ) yield (vds, groups)

    forAll(g) { case (vds, groups) =>
      val path1 = tmpDir.createTempFile("file1", ".vds")
      val path2 = tmpDir.createTempFile("file2", ".vds")
      val path3 = tmpDir.createTempFile("file3", ".vds")

      vds.filterVariants { case (v, _, _) => groups(v) == 1 }
        .write(sqlContext, path1)

      vds.filterVariants { case (v, _, _) => groups(v) == 2 }
        .write(sqlContext, path2)

      vds.filterVariants { case (v, _, _) => groups(v) == 3 }
        .write(sqlContext, path3)

      Read.run(State(sc, sqlContext, null), Array(path1, path2, path3))
        .vds.same(vds)

    }.check()
  }
}
