package is.hail.methods

import breeze.linalg.{Vector => BVector}
import is.hail.SparkSuite
import is.hail.annotations.{Annotation, Region, RegionValue, RegionValueBuilder}
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.expr.types._
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import org.testng.annotations.Test

case class BitPackedVector(gs: Array[Long], nSamples: Int, mean: Double, stdDevRec: Double) {
  def unpack(): Array[Int] = {
    val gts = Array.ofDim[Int](nSamples)
    val nPacks = gs.length

    var packIndex = 0
    var i = 0
    val shiftInit = LDPrune.genotypesPerPack * 2 - 2
    while (packIndex < nPacks && i < nSamples) {
      val l = gs(packIndex)
      var shift = shiftInit
      while (shift >= 0 && i < nSamples) {
        val gt = (l >> shift) & 3
        if (gt == 3)
          gts(i) = -1
        else
          gts(i) = gt.toInt
        shift -= 2
        i += 1
      }
      packIndex += 1
    }

    gts
  }
}

object LDPruneSuite {
  val rowType = TStruct(
    "pk" -> GenomeReference.GRCh37.locusType,
    "v" -> GenomeReference.GRCh37.variantType,
    "va" -> TStruct(),
    "gs" -> TArray(Genotype.htsGenotypeType)
  )

  val bitPackedVectorViewType = BitPackedVectorView.rowType(rowType.fieldType(0), rowType.fieldType(1))

  def makeRV(gs: Iterable[Annotation]): RegionValue = {
    val gArr = gs.toIndexedSeq
    val rvb = new RegionValueBuilder(Region())
    rvb.start(rowType)
    rvb.startStruct()
    rvb.setMissing()
    rvb.setMissing()
    rvb.setMissing()
    rvb.addAnnotation(TArray(Genotype.htsGenotypeType), gArr)
    rvb.endStruct()
    rvb.end()
    rvb.result()
  }

  def convertGtsToGs(gts: Array[Int]): Iterable[Annotation] = gts.map(Genotype(_)).toIterable

  def toBitPackedVectorView(gs: HailIterator[Int], nSamples: Int): Option[BitPackedVectorView] =
    toBitPackedVectorView(convertGtsToGs(gs.toArray), nSamples)

  def toBitPackedVectorView(gs: Iterable[Annotation], nSamples: Int): Option[BitPackedVectorView] = {
    val bpvv = new BitPackedVectorView(bitPackedVectorViewType)
    toBitPackedVectorRegionValue(gs, nSamples) match {
      case Some(rv) =>
        bpvv.setRegion(rv)
        Some(bpvv)
      case None => None
    }
  }

  def toBitPackedVectorRegionValue(gs: Iterable[Annotation], nSamples: Int): Option[RegionValue] = {
    toBitPackedVectorRegionValue(makeRV(gs), nSamples)
  }

  def toBitPackedVectorRegionValue(rv: RegionValue, nSamples: Int): Option[RegionValue] = {
    val rvb = new RegionValueBuilder(Region())
    val hcView = HardCallView(rowType)
    hcView.setRegion(rv)

    rvb.start(bitPackedVectorViewType)
    rvb.startStruct()
    rvb.setMissing()
    rvb.setMissing()
    val keep = LDPrune.addBitPackedVector(rvb, hcView, nSamples)

    if (keep) {
      rvb.endStruct()
      rvb.end()
      Some(rvb.result())
    }
    else
      None
  }

  def toBitPackedVector(gts: Array[Int]): Option[BitPackedVector] = {
    val nSamples = gts.length
    toBitPackedVectorView(convertGtsToGs(gts), nSamples).map { bpvv =>
      BitPackedVector((0 until bpvv.getNPacks).map(bpvv.getPack).toArray, bpvv.getNSamples, bpvv.getMean, bpvv.getStdDevRecip)
    }
  }
}

class LDPruneSuite extends SparkSuite {
  val bytesPerCoreMB = 256
  val nCores = 4

  def correlationMatrix(gts: Array[Iterable[Annotation]], nSamples: Int) = {
    val bvi = gts.map { gs => LDPruneSuite.toBitPackedVectorView(gs, nSamples) }
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

  def isUncorrelated(vds: MatrixTable, r2Threshold: Double, windowSize: Int): Boolean = {
    val nSamplesLocal = vds.nSamples
    val r2Matrix = correlationMatrix(vds.typedRDD[Locus, Variant].map { case (v, (va, gs)) => gs }.collect(), nSamplesLocal)
    val variantMap = vds.variants.zipWithIndex().map { case (v, i) => (i.toInt, v.asInstanceOf[Variant]) }.collectAsMap()

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
      assert(LDPruneSuite.toBitPackedVector(gts).forall { bpv =>
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

    val computedR2 = correlationMatrix(gts.map(LDPruneSuite.convertGtsToGs), 8)

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
    val bvi1 = LDPruneSuite.toBitPackedVectorView(LDPruneSuite.convertGtsToGs(input), input.length).get
    val bvi2 = LDPruneSuite.toBitPackedVectorView(LDPruneSuite.convertGtsToGs(input), input.length).get

    assert(D_==(LDPrune.computeR2(bvi1, bvi2), 1d))
  }

  @Test def testIdenticalVariants() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/ldprune2.vcf", nPartitions = Option(2)))
    val prunedVds = LDPrune(vds, nCores, 0.2, 700, memoryPerCoreMB = bytesPerCoreMB)
    assert(prunedVds.countVariants() == 1)
  }

  @Test def testMultipleChr() = {
    val r2 = 0.2
    val windowSize = 500
    val vds = SplitMulti(hc.importVCF("src/test/resources/ldprune_multchr.vcf", nPartitions = Option(10)))
    val prunedVds = LDPrune(vds, nCores, r2, windowSize, bytesPerCoreMB)
    assert(isUncorrelated(prunedVds, r2, windowSize))
  }

  object Spec extends Properties("LDPrune") {
    val compGen = for (r2: Double <- Gen.choose(0.5, 1.0);
      windowSize: Int <- Gen.choose(0, 5000);
      nPartitions: Int <- Gen.choose(5, 10)) yield (r2, windowSize, nPartitions)

    val vectorGen = for (nSamples: Int <- Gen.choose(1, 1000);
      v1: Array[Int] <- Gen.buildableOfN[Array](nSamples, Gen.choose(-1, 2));
      v2: Array[Int] <- Gen.buildableOfN[Array](nSamples, Gen.choose(-1, 2))
    ) yield (nSamples, v1, v2)

    property("bitPacked pack and unpack give same as orig") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[Int], _) =>
        val bpv = LDPruneSuite.toBitPackedVector(v1)

        bpv match {
          case Some(x) => LDPruneSuite.toBitPackedVector(x.unpack()).get.gs sameElements x.gs
          case None => true
        }
      }

    property("R2 bitPacked same as BVector") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[Int], v2: Array[Int]) =>
        val v1Ann = LDPruneSuite.convertGtsToGs(v1)
        val v2Ann = LDPruneSuite.convertGtsToGs(v2)

        val bv1 = LDPruneSuite.toBitPackedVectorView(v1Ann, nSamples)
        val bv2 = LDPruneSuite.toBitPackedVectorView(v2Ann, nSamples)

        val view = HardCallView(LDPruneSuite.rowType)

        val rv1 = LDPruneSuite.makeRV(v1Ann)
        view.setRegion(rv1)
        val sgs1 = RegressionUtils.normalizedHardCalls(view, nSamples).map(math.sqrt(1d / nSamples) * BVector(_))

        val rv2 = LDPruneSuite.makeRV(v2Ann)
        view.setRegion(rv2)
        val sgs2 = RegressionUtils.normalizedHardCalls(view, nSamples).map(math.sqrt(1d / nSamples) * BVector(_))

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
        val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Option(nPartitions)))
        val prunedVds = LDPrune(vds, nCores, r2, windowSize, bytesPerCoreMB)
        isUncorrelated(prunedVds, r2, windowSize)
      }
  }

  @Test def testRandom() {
    Spec.check()
  }

  @Test def testInputs() {
    def vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Option(10)))

    // memory per core requirement
    intercept[HailException](LDPrune(vds, nCores, r2Threshold = 0.2, windowSize = 1000, memoryPerCoreMB = 0))

    // r2 negative
    intercept[HailException](LDPrune(vds, nCores, r2Threshold = -0.1, windowSize = 1000))

    // r2 > 1
    intercept[HailException](LDPrune(vds, nCores, r2Threshold = 1.1, windowSize = 1000))

    // windowSize negative
    intercept[HailException](LDPrune(vds, nCores, r2Threshold = 0.5, windowSize = -2))

    // parallelism negative
    intercept[HailException](LDPrune(vds, nCores = -1, r2Threshold = 0.5, windowSize = 100))
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
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf.bgz"))
    val prunedVds = LDPrune(vds, nCores, r2Threshold = 0.2, windowSize = 100000, memoryPerCoreMB = 200)
    assert(isUncorrelated(prunedVds, 0.2, 1000))
  }

  @Test def testNoPrune() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf.bgz"))
    val nSamples = vds.nSamples
    val filteredVDS = vds.copyLegacy(
      genotypeSignature = Genotype.htsGenotypeType,
      rdd = vds.typedRDD[Locus, Variant]
        .filter { case (v, (va, gs)) =>
          v.isBiallelic &&
            LDPruneSuite.toBitPackedVectorView(gs.hardCallIterator, nSamples).isDefined
        })
    val prunedVDS = LDPrune(filteredVDS, nCores, r2Threshold = 1, windowSize = 0, memoryPerCoreMB = 200)
    assert(prunedVDS.countVariants() == filteredVDS.countVariants())
  }
}
