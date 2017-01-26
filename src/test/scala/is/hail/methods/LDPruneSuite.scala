package is.hail.methods

import breeze.linalg.{Vector => BVector}
import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.driver._
import is.hail.variant._
import is.hail.utils._
import org.testng.annotations.Test

class LDPruneSuite extends SparkSuite {
  val bytesPerCore = 256L * 1024L * 1024L

  def correlationMatrixBitPacked(gs: Array[Iterable[Genotype]], nSamples: Int) = {
    val bvi = gs.map(LDPrune.toByteGtArray(_, nSamples))
    val r2 = for (i <- bvi.indices; j <- bvi.indices) yield {
      (bvi(i), bvi(j)) match {
        case (Some(x), Some(y)) =>
          Some(LDPrune.r2(x, y))
        case _ => None
      }
    }
    val nVariants = bvi.length
    new MultiArray2(nVariants, nVariants, r2.toArray)
  }

  def correlationMatrix(gs: Array[Iterable[Genotype]], nSamples: Int) = {
    val sgs = gs.map(LDPrune.toNormalizedGtArray(_, nSamples))
    val r2 = for (i <- sgs.indices; j <- sgs.indices) yield {
      (sgs(i), sgs(j)) match {
        case (Some(x), Some(y)) =>
          val r = BVector(x).dot(BVector(y)): Double
          Some(r * r)
        case _ => None
      }
    }
    val nVariants = sgs.length
    new MultiArray2(nVariants, nVariants, r2.toArray)
  }

  def uncorrelated(vds: VariantDataset, r2Threshold: Double, window: Int): Boolean = {
    val nSamplesLocal = vds.nSamples
    val r2Matrix = correlationMatrix(vds.rdd.map { case (v, (va, gs)) => gs }.collect(), nSamplesLocal)
    val variantMap = vds.variants.zipWithIndex().map { case (v, i) => (i.toInt, v) }.collectAsMap()

    r2Matrix.indices.forall { case (i, j) =>
      val v1 = variantMap(i)
      val v2 = variantMap(j)
      val r2 = r2Matrix(i, j)

      v1 == v2 ||
        v1.contig != v2.contig ||
        (v1.contig == v2.contig && math.abs(v1.start - v2.start) > window) ||
        r2.exists(_ < r2Threshold)
    }
  }

  @Test def testR2() {
    val gts = Array(
      Array(1, 0, 0, 0, 0, 0, 0, 0),
      Array(1, 1, 1, 1, 1, 1, 1, 1),
      Array(1, 2, 2, 2, 2, 2, 2, 2),
      Array(1, 0, 0, 0, 1, 1, 1, 1),
      Array(1, 0, 0, 0, 1, 1, 2, 2),
      Array(1, 0, 1, 1, 2, 2, 0, 1),
      Array(1, 0, 1, 0, 2, 2, 1, 1)
    )

    val actualR2 = new MultiArray2(7, 7, hadoopConf.readLines("src/test/resources/ldprune_corrtest.txt")(_.flatMap(_.map { line =>
      line.trim.split("\t").map(r2 => if (r2 == "NA") None else Some(r2.toDouble))
    }.value).toArray))

    val computedR2 = correlationMatrix(gts.map(_.map(Genotype(_)).toIterable), 8)
    val computedR2bitpacked = correlationMatrixBitPacked(gts.map(_.map(Genotype(_)).toIterable), 8)

    val res = actualR2.indices.forall { case (i, j) =>
      val expected = actualR2(i, j)
      val computed = computedR2bitpacked(i, j)
//      val computed = computedR2(i, j)

      (computed, expected) match {
        case (Some(x), Some(y)) =>
          val res = math.abs(x - y) < 1e-6
          if (!res)
            info(s"i=$i j=$j r2Computed=$x r2Expected=$y")
          res
        case (None, None) => true
        case _ =>
          info(s"i=$i j=$j r2Computed=$computed r2Expected=$expected")
          false
      }
    }

    assert(res)
  }

  @Test def testIdenticalVariants() {
    var s = State(sc, sqlContext, null)
    s = ImportVCF.run(s, Array("-i", "src/test/resources/ldprune2.vcf", "-n", "2"))
    s = SplitMulti.run(s, Array.empty[String])
    val prunedVds = LDPrune.ldPrune(s.vds, 0.2, 700, bytesPerCore)
    assert(prunedVds.nVariants == 1)
  }

  @Test def testMultipleChr() = {
    val r2 = 0.2
    val window = 500

    var s = State(sc, sqlContext, null)
    s = ImportVCF.run(s, Array("-i", "src/test/resources/ldprune_multchr.vcf", "-n", "10"))
    s = SplitMulti.run(s, Array.empty[String])
    val prunedVds = LDPrune.ldPrune(s.vds, r2, window, bytesPerCore)

    assert(uncorrelated(prunedVds, r2, window))
  }

  object Spec extends Properties("LDPrune") {
    val compGen = for (r2: Double <- Gen.choose(0.5, 1.0);
      window: Int <- Gen.choose(0, 5000);
      numPartitions: Int <- Gen.choose(5, 10)) yield (r2, window, numPartitions)

    property("uncorrelated") =
      forAll(compGen) { case (r2: Double, window: Int, numPartitions: Int) =>
        var s = State(sc, sqlContext, null)
        s = ImportVCF.run(s, Array("-i", "src/test/resources/sample.vcf.bgz", "-n", s"$numPartitions"))
        s = SplitMulti.run(s, Array.empty[String])
        val prunedVds = LDPrune.ldPrune(s.vds, r2, window, bytesPerCore)
        uncorrelated(prunedVds, r2, window)
      }
  }

  @Test def testRandom() {
    Spec.check()
  }

  @Test def testInputs() {
    def setup() = {
      var s = State(sc, sqlContext, null)
      s = ImportVCF.run(s, Array("-i", "src/test/resources/sample.vcf.bgz", "-n", "10"))
      SplitMulti.run(s, Array.empty[String])
    }

    // memory per core requirement
    intercept[FatalException] {
      val s = setup()
      s.copy(vds = LDPrune.ldPrune(s.vds, 0.2, 1000, 0))
    }

    // r2 negative
    intercept[FatalException] {
      val s = setup()
      s.copy(vds = LDPrune.ldPrune(s.vds, -0.1, 1000, 1000))
    }

    // r2 > 1
    intercept[FatalException] {
      val s = setup()
      val prunedVds = LDPrune.ldPrune(s.vds, 1.1, 1000, 1000)
    }

    // window negative
    intercept[FatalException] {
      val s = setup()
      s.copy(vds = LDPrune.ldPrune(s.vds, 0.5, -2, 1000))
    }
  }

  @Test def testMemoryRequirements() {
    val a = LDPrune.estimateMemoryRequirements(nVariants = 1, nSamples = 1, memoryPerCore = 512 * 1024 * 1024)
    assert(a._2 == 1)

    val nSamples = 5
    val nVariants = 5
    val memoryPerVariant = LDPrune.variantByteOverhead + 8 * nSamples
    val recipFractionMemoryUsed = 1.0 / LDPrune.fractionMemoryToUse
    val memoryPerCore = math.ceil(memoryPerVariant * recipFractionMemoryUsed).toInt

    for (i <- 1 to nVariants) {
      val y = LDPrune.estimateMemoryRequirements(nVariants, nSamples, memoryPerCore * i)
      assert(y._1 == i && y._2 == math.ceil(nVariants.toDouble / i).toInt)
    }
  }

  @Test def testWindow() {
    var s = State(sc, sqlContext, null)
    s = ImportVCF.run(s, Array("-i", "src/test/resources/sample.vcf.bgz"))
    s = SplitMulti.run(s, Array.empty[String])
    val prunedVds = LDPrune.ldPrune(s.vds, 0.2, 100000, 200000)
    assert(uncorrelated(prunedVds, 0.2, 1000))
  }
}
