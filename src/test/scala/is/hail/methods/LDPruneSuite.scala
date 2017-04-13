package is.hail.methods

import breeze.linalg.{Vector => BVector}
import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import org.testng.annotations.Test

class LDPruneSuite extends SparkSuite {
  val bytesPerCore = 256L * 1024L * 1024L

  def convertGtsToGs(gts: Array[Int]): Iterable[Genotype] = gts.map(Genotype(_)).toIterable

  def correlationMatrix(gts: Array[Iterable[Genotype]], nSamples: Int) = {
    val bvi = gts.map { gs => LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples) }
    val r2 = for (i <- bvi.indices; j <- bvi.indices) yield {
      (bvi(i), bvi(j)) match {
        case (Some(x), Some(y)) =>
          Some(LDPrune.computeR2(x, y))
        case _ => None
      }
    }
    val nVariants = bvi.length
    new MultiArray2(nVariants, nVariants, r2.toArray)
  }

  def isUncorrelated(vds: VariantDataset, r2Threshold: Double, windowSize: Int): Boolean = {
    val nSamplesLocal = vds.nSamples
    val r2Matrix = correlationMatrix(vds.rdd.map { case (v, (va, gs)) => gs }.collect(), nSamplesLocal)
    val variantMap = vds.variants.zipWithIndex().map { case (v, i) => (i.toInt, v) }.collectAsMap()

    r2Matrix.indices.forall { case (i, j) =>
      val v1 = variantMap(i)
      val v2 = variantMap(j)
      val r2 = r2Matrix(i, j)

      v1 == v2 ||
        v1.contig != v2.contig ||
        (v1.contig == v2.contig && math.abs(v1.start - v2.start) > windowSize) ||
        r2.exists(_ < r2Threshold)
    }
  }

  @Test def testBitPackUnpack() {
    val gts1 = Array(-1, 0, 1, 2, 1, 1, 0, 0, 0, 0, 2, 2, -1, -1, -1, -1)
    val gts2 = Array(0, 1, 2, 2, 2, 0, -1, -1)
    val gts3 = gts1 ++ Array.ofDim[Int](32 - gts1.length) ++ gts2

    for (gts <- Array(gts1, gts2, gts3)) {
      val n = gts.length
      val gs = convertGtsToGs(gts)
      assert(LDPrune.toBitPackedVector(gs.hardCallIterator, n).forall { bpv =>
        bpv.unpack() sameElements gts
      })
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

    val computedR2 = correlationMatrix(gts.map(convertGtsToGs), 8)

    val isSame = actualR2.indices.forall { case (i, j) =>
      val expected = actualR2(i, j)
      val computed = computedR2(i, j)

      (computed, expected) match {
        case (Some(x), Some(y)) =>
          val isSame = D_==(x, y)
          if (!isSame)
            info(s"i=$i j=$j r2Computed=$x r2Expected=$y")
          isSame
        case (None, None) => true
        case _ =>
          info(s"i=$i j=$j r2Computed=$computed r2Expected=$expected")
          false
      }
    }

    assert(isSame)

    val input = Array(0, 1, 2, 2, 2, 0, -1, -1)
    val gs = convertGtsToGs(input)
    val n = input.length
    val bvi1 = LDPrune.toBitPackedVector(gs.hardCallIterator, n).get
    val bvi2 = LDPrune.toBitPackedVector(gs.hardCallIterator, n).get

    assert(D_==(LDPrune.computeR2(bvi1, bvi2), 1d))
  }

  @Test def testIdenticalVariants() {
    val vds = hc.importVCF("src/test/resources/ldprune2.vcf", nPartitions = Option(2)).splitMulti()
    val prunedVds = LDPrune(vds, 0.2, 700, nCores = 4, memoryPerCore = bytesPerCore)
    assert(prunedVds.countVariants() == 1)
  }

  @Test def testMultipleChr() = {
    val r2 = 0.2
    val windowSize = 500
    val vds = hc.importVCF("src/test/resources/ldprune_multchr.vcf", nPartitions = Option(10)).splitMulti()
    val prunedVds = LDPrune(vds, r2, windowSize, nCores = 4, memoryPerCore = bytesPerCore)
    assert(isUncorrelated(prunedVds, r2, windowSize))
  }

  object Spec extends Properties("LDPrune") {
    val compGen = for (r2: Double <- Gen.choose(0.5, 1.0);
      windowSize: Int <- Gen.choose(0, 5000);
      nPartitions: Int <- Gen.choose(5, 10)) yield (r2, windowSize, nPartitions)

    val vectorGen = for (nSamples: Int <- Gen.choose(1, 1000);
      v1: Array[Int] <- Gen.buildableOfN[Array, Int](nSamples, Gen.choose(-1, 2));
      v2: Array[Int] <- Gen.buildableOfN[Array, Int](nSamples, Gen.choose(-1, 2))
    ) yield (nSamples, v1, v2)

    property("bitPacked pack and unpack give same as orig") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[Int], v2: Array[Int]) =>
        val gs1 = convertGtsToGs(v1)
        val gs2 = convertGtsToGs(v2)
        val bv1 = LDPrune.toBitPackedVector(gs1.hardCallIterator, nSamples)
        val bv2 = LDPrune.toBitPackedVector(gs2.hardCallIterator, nSamples)

        val isSame = (bv1, bv2) match {
          case (Some(x), Some(y)) =>
            (LDPrune.toBitPackedVector(convertGtsToGs(x.unpack()).hardCallIterator, nSamples).get.gs sameElements bv1.get.gs) &&
              (LDPrune.toBitPackedVector(convertGtsToGs(y.unpack()).hardCallIterator, nSamples).get.gs sameElements bv2.get.gs)
          case _ => true
        }
        isSame
      }

    property("R2 bitPacked same as BVector") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[Int], v2: Array[Int]) =>
        val gs1 = convertGtsToGs(v1)
        val gs2 = convertGtsToGs(v2)
        val bv1 = LDPrune.toBitPackedVector(gs1.hardCallIterator, nSamples)
        val bv2 = LDPrune.toBitPackedVector(gs2.hardCallIterator, nSamples)
        val sgs1 = RegressionUtils.toNormalizedGtArray(gs1, nSamples).map(math.sqrt(1d / nSamples) * BVector(_))
        val sgs2 = RegressionUtils.toNormalizedGtArray(gs2, nSamples).map(math.sqrt(1d / nSamples) * BVector(_))

        (bv1, bv2, sgs1, sgs2) match {
          case (Some(a), Some(b), Some(c: BVector[Double]), Some(d: BVector[Double])) =>
            val rBreeze = c.dot(d): Double
            val r2Breeze = rBreeze * rBreeze
            val r2BitPacked = LDPrune.computeR2(a, b)

            val isSame = D_==(r2BitPacked, r2Breeze) && D_>=(r2BitPacked, 0d) && D_<=(r2BitPacked, 1d)
            if (!isSame) {
              println(s"breeze=$r2Breeze bitPacked=$r2BitPacked nSamples=$nSamples")
            }
            isSame
          case _ => true
        }
      }

    property("uncorrelated") =
      forAll(compGen) { case (r2: Double, windowSize: Int, nPartitions: Int) =>
        val vds = hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Option(nPartitions)).splitMulti()
        val prunedVds = LDPrune(vds, r2, windowSize, nCores = 4, memoryPerCore = bytesPerCore)
        isUncorrelated(prunedVds, r2, windowSize)
      }
  }

  @Test def testRandom() {
    Spec.check()
  }

  @Test def testInputs() {
    def vds = {
      hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Option(10)).splitMulti()
    }

    // memory per core requirement
    intercept[HailException] {
      val prunedVds = LDPrune(vds, 0.2, 1000, nCores = 1, memoryPerCore = 0)
    }

    // r2 negative
    intercept[HailException] {
      val prunedVds = LDPrune(vds, -0.1, 1000, nCores = 1, memoryPerCore = 1000)
    }

    // r2 > 1
    intercept[HailException] {
      val prunedVds = LDPrune(vds, 1.1, 1000, nCores = 1, memoryPerCore = 1000)
    }

    // windowSize negative
    intercept[HailException] {
      val prunedVds = LDPrune(vds, 0.5, -2, nCores = 1, memoryPerCore = 1000)
    }

    // parallelism negative
    intercept[HailException] {
      val prunedVds = LDPrune(vds, 0.5, -2, nCores = -1, memoryPerCore = 1000)
    }
  }

  @Test def testMemoryRequirements() {
    val nSamples = 5
    val nVariants = 100
    val memoryPerVariant = LDPrune.variantByteOverhead + math.ceil(8 * nSamples.toDouble / LDPrune.genotypesPerPack).toLong
    val recipFractionMemoryUsed = 1.0 / LDPrune.fractionMemoryToUse
    val memoryPerCore = math.ceil(memoryPerVariant * recipFractionMemoryUsed).toInt

    for (maxQueueSize <- 1 to nVariants; nCores <- 1 to 5) {
      val y = LDPrune.estimateMemoryRequirements(nVariants, nSamples, nCores = nCores, memoryPerCore * maxQueueSize)
      assert(y._1 == maxQueueSize && y._2 == math.max(nCores * LDPrune.nPartitionsPerCore, math.ceil(nVariants.toDouble / maxQueueSize).toInt))
    }
  }

  @Test def testWindow() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz").splitMulti()
    val prunedVds = LDPrune(vds, 0.2, 100000, nCores = 4, memoryPerCore = 200000)
    assert(isUncorrelated(prunedVds, 0.2, 1000))
  }

  @Test def testNoPrune() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
      .splitMulti()
    val nSamples = vds.nSamples
    val filteredVds = vds.filterVariants{ case (v, va, gs) => v.isBiallelic && LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples).isDefined }
    val prunedVds = LDPrune(filteredVds, 1, 0, nCores = 4, memoryPerCore = 200000)
    assert(prunedVds.countVariants() == filteredVds.countVariants())
  }
}
