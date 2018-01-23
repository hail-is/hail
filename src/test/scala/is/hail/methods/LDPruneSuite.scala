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
import is.hail.testUtils._
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
  val rvRowType = TStruct(
    "locus" -> GenomeReference.GRCh37.locusType,
    "alleles" -> TArray(TString()),
    MatrixType.entriesIdentifier -> TArray(Genotype.htsGenotypeType)
  )

  val bitPackedVectorViewType = BitPackedVectorView.rvRowType(rvRowType.fieldByName("locus").typ,
    rvRowType.fieldByName("alleles").typ)

  def makeRV(gs: Iterable[Annotation]): RegionValue = {
    val gArr = gs.toIndexedSeq
    val rvb = new RegionValueBuilder(Region())
    rvb.start(rvRowType)
    rvb.startStruct()
    rvb.addAnnotation(rvRowType.fieldType(0), Locus("1", 1))
    rvb.addAnnotation(rvRowType.fieldType(1), IndexedSeq("A", "T"))
    rvb.addAnnotation(TArray(Genotype.htsGenotypeType), gArr)
    rvb.endStruct()
    rvb.end()
    rvb.result()
  }

  def convertCallsToGs(calls: Array[BoxedCall]): Iterable[Annotation] = calls.map(Genotype(_)).toIterable

  // expecting iterable of Genotype with htsjdk schema
  def toBitPackedVectorView(gs: Iterable[Annotation], nSamples: Int): Option[BitPackedVectorView] = {
    val bpvv = new BitPackedVectorView(bitPackedVectorViewType)
    toBitPackedVectorRegionValue(gs, nSamples) match {
      case Some(rv) =>
        bpvv.setRegion(rv)
        Some(bpvv)
      case None => None
    }
  }

  // expecting iterable of Genotype with htsjdk schema
  def toBitPackedVectorRegionValue(gs: Iterable[Annotation], nSamples: Int): Option[RegionValue] = {
    toBitPackedVectorRegionValue(makeRV(gs), nSamples)
  }

  def toBitPackedVectorRegionValue(rv: RegionValue, nSamples: Int): Option[RegionValue] = {
    val rvb = new RegionValueBuilder(Region())
    val hcView = HardCallView(rvRowType)
    hcView.setRegion(rv)

    rvb.start(bitPackedVectorViewType)
    rvb.startStruct()
    rvb.addAnnotation(rvRowType.fieldType(0), Locus("1", 1))
    rvb.addAnnotation(rvRowType.fieldType(1), IndexedSeq("A", "T"))
    val keep = LDPrune.addBitPackedVector(rvb, hcView, nSamples)

    if (keep) {
      rvb.endStruct()
      rvb.end()
      Some(rvb.result())
    }
    else
      None
  }

  def toBitPackedVector(calls: Array[BoxedCall]): Option[BitPackedVector] = {
    val nSamples = calls.length
    toBitPackedVectorView(convertCallsToGs(calls), nSamples).map { bpvv =>
      BitPackedVector((0 until bpvv.getNPacks).map(bpvv.getPack).toArray, bpvv.getNSamples, bpvv.getMean, bpvv.getStdDevRecip)
    }
  }
}

class LDPruneSuite extends SparkSuite {
  val bytesPerCoreMB = 256
  val nCores = 4

  def toC2(i: Int): BoxedCall = if (i == -1) null else Call2.fromUnphasedDiploidGtIndex(i)

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
    val nSamplesLocal = vds.numCols
    val r2Matrix = correlationMatrix(vds.typedRDD[Variant].map { case (v, (va, gs)) => gs }.collect(), nSamplesLocal)
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
    val calls1 = Array(-1, 0, 1, 2, 1, 1, 0, 0, 0, 0, 2, 2, -1, -1, -1, -1).map(toC2)
    val calls2 = Array(0, 1, 2, 2, 2, 0, -1, -1).map(toC2)
    val calls3 = calls1 ++ Array.ofDim[Int](32 - calls1.length).map(toC2) ++ calls2

    for (calls <- Array(calls1, calls2, calls3)) {
      assert(LDPruneSuite.toBitPackedVector(calls).forall { bpv =>
        bpv.unpack().map(toC2) sameElements calls
      })
    }
  }

  @Test def testR2() {
    val calls = Array(
      Array(1, 0, 0, 0, 0, 0, 0, 0).map(toC2),
      Array(1, 1, 1, 1, 1, 1, 1, 1).map(toC2),
      Array(1, 2, 2, 2, 2, 2, 2, 2).map(toC2),
      Array(1, 0, 0, 0, 1, 1, 1, 1).map(toC2),
      Array(1, 0, 0, 0, 1, 1, 2, 2).map(toC2),
      Array(1, 0, 1, 1, 2, 2, 0, 1).map(toC2),
      Array(1, 0, 1, 0, 2, 2, 1, 1).map(toC2)
    )

    val actualR2 = new MultiArray2(7, 7, hadoopConf.readLines("src/test/resources/ldprune_corrtest.txt")(_.flatMap(_.map { line =>
      line.trim.split("\t").map(r2 => if (r2 == "NA") None else Some(r2.toDouble))
    }.value).toArray))

    val computedR2 = correlationMatrix(calls.map(LDPruneSuite.convertCallsToGs), 8)

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

    val input = Array(0, 1, 2, 2, 2, 0, -1, -1).map(toC2)
    val bvi1 = LDPruneSuite.toBitPackedVectorView(LDPruneSuite.convertCallsToGs(input), input.length).get
    val bvi2 = LDPruneSuite.toBitPackedVectorView(LDPruneSuite.convertCallsToGs(input), input.length).get

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
      v1: Array[BoxedCall] <- Gen.buildableOfN[Array](nSamples, Gen.choose(-1, 2).map(toC2));
      v2: Array[BoxedCall] <- Gen.buildableOfN[Array](nSamples, Gen.choose(-1, 2).map(toC2))
    ) yield (nSamples, v1, v2)

    property("bitPacked pack and unpack give same as orig") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[BoxedCall], _) =>
        val bpv = LDPruneSuite.toBitPackedVector(v1)

        bpv match {
          case Some(x) => LDPruneSuite.toBitPackedVector(x.unpack().map(toC2)).get.gs sameElements x.gs
          case None => true
        }
      }

    property("R2 bitPacked same as BVector") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[BoxedCall], v2: Array[BoxedCall]) =>
        val v1Ann = LDPruneSuite.convertCallsToGs(v1)
        val v2Ann = LDPruneSuite.convertCallsToGs(v2)

        val bv1 = LDPruneSuite.toBitPackedVectorView(v1Ann, nSamples)
        val bv2 = LDPruneSuite.toBitPackedVectorView(v2Ann, nSamples)

        val view = HardCallView(LDPruneSuite.rvRowType)

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
    val nSamples = vds.numCols
    val filteredVDS = vds.filterVariantsExpr("gs.filter(g => isDefined(g.GT)).map(_ => g.GT).collectAsSet().size() > 1")
    val prunedVDS = LDPrune(filteredVDS, nCores, r2Threshold = 1, windowSize = 0, memoryPerCoreMB = 200)
    assert(prunedVDS.countVariants() == filteredVDS.countVariants())
  }
}
