package is.hail.methods

import is.hail.{HailSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.{Gen, Properties}
import is.hail.check.Prop._
import is.hail.expr.ir.{Interpret, MatrixValue, TableValue}
import is.hail.utils._
import is.hail.variant._

import breeze.linalg.{Vector => BVector}
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

object LocalLDPruneSuite {
  val variantByteOverhead = 50
  val fractionMemoryToUse = 0.25

  def fromCalls(calls: IndexedSeq[BoxedCall]): Option[BitPackedVector] = {
    val locus = Locus("1", 1)
    val alleles = Array("A", "T")
    val nSamples = calls.length
    val builder = new BitPackedVectorBuilder(nSamples)
    for (call <- calls)
      if (call == null)
        builder.addMissing()
      else
        builder.addGT(call)

    Option(builder.finish(locus, alleles))
  }

  def correlationMatrixGT(gts: Array[Iterable[Annotation]]) = correlationMatrix(gts.map { gts =>
    gts.map(gt => Genotype.call(gt).map(c => c: BoxedCall).orNull).toArray
  })

  def correlationMatrix(gts: Array[Array[BoxedCall]]) = {
    val bvi = gts.map(gs => LocalLDPruneSuite.fromCalls(gs.toIndexedSeq))
    val r2 =
      for {
        i <- bvi.indices
        j <- bvi.indices
      } yield (bvi(i), bvi(j)) match {
        case (Some(x), Some(y)) =>
          Some(LocalLDPrune.computeR2(x, y))
        case _ => None
      }
    val nVariants = bvi.length
    new MultiArray2(nVariants, nVariants, r2.toArray)
  }

  def estimateMemoryRequirements(nVariants: Long, nSamples: Int, memoryPerCore: Long): Int = {
    val bytesPerVariant = math.ceil(
      8 * nSamples.toDouble / BitPackedVector.GENOTYPES_PER_PACK
    ).toLong + variantByteOverhead
    val memoryAvailPerCore = memoryPerCore * fractionMemoryToUse

    val maxQueueSize = math.max(1, math.ceil(memoryAvailPerCore / bytesPerVariant).toInt)
    maxQueueSize
  }

  def normalizedHardCalls(calls: Array[BoxedCall]): Option[Array[Double]] = {
    val nSamples = calls.length
    val vals = Array.ofDim[Double](nSamples)
    var nMissing = 0
    var sum = 0
    var sumSq = 0

    for ((call, i) <- calls.zipWithIndex) {
      if (call != null) {
        val gt = Call.unphasedDiploidGtIndex(call)
        vals(i) = gt
        (gt: @unchecked) match {
          case 0 =>
          case 1 =>
            sum += 1
            sumSq += 1
          case 2 =>
            sum += 2
            sumSq += 4
        }
      } else {
        vals(i) = -1
        nMissing += 1
      }
    }

    val nPresent = nSamples - nMissing
    val nonConstant = !(sum == 0 || sum == 2 * nPresent || sum == nPresent && sumSq == nPresent)

    if (nonConstant) {
      val mean = sum.toDouble / nPresent
      val meanSq = (sumSq + nMissing * mean * mean) / nSamples
      val stdDev = math.sqrt(meanSq - mean * mean)

      val gtDict = Array(0, -mean / stdDev, (1 - mean) / stdDev, (2 - mean) / stdDev)
      var i = 0
      while (i < nSamples) {
        vals(i) = gtDict(vals(i).toInt + 1)
        i += 1
      }

      Some(vals)
    } else
      None
  }
}

class LocalLDPruneSuite extends HailSuite {
  val memoryPerCoreBytes = 256 * 1024 * 1024
  val nCores = 4

  lazy val mt = Interpret(
    TestUtils.importVCF(ctx, "src/test/resources/sample.vcf.bgz", nPartitions = Option(10)),
    ctx,
    false,
  ).toMatrixValue(Array("s"))

  lazy val maxQueueSize = LocalLDPruneSuite.estimateMemoryRequirements(
    mt.rvd.count(),
    mt.nCols,
    memoryPerCoreBytes,
  )

  def toC2(i: Int): BoxedCall = if (i == -1) null else Call2.fromUnphasedDiploidGtIndex(i)

  def getLocallyPrunedRDDWithGT(
    unprunedMatrixTable: MatrixValue,
    locallyPrunedTable: TableValue,
  ): RDD[(Locus, Any, Iterable[Annotation])] = {
    val mtLocusIndex = unprunedMatrixTable.rvRowPType.index("locus").get
    val mtAllelesIndex = unprunedMatrixTable.rvRowPType.index("alleles").get
    val mtEntriesIndex = unprunedMatrixTable.entriesIdx

    val locusIndex = locallyPrunedTable.rvd.rowType.fieldIdx("locus")
    val allelesIndex = locallyPrunedTable.rvd.rowType.fieldIdx("alleles")

    val locallyPrunedVariants = locallyPrunedTable.rdd.mapPartitions(
      it => it.map(row => (row.get(locusIndex), row.get(allelesIndex))),
      preservesPartitioning = true,
    ).collectAsSet()

    unprunedMatrixTable.rvd.toRows.map { r =>
      (
        r.getAs[Locus](mtLocusIndex),
        r.getAs[Any](mtAllelesIndex),
        r.getAs[Iterable[Annotation]](mtEntriesIndex),
      )
    }.filter { case (locus, alleles, _) => locallyPrunedVariants.contains((locus, alleles)) }
  }

  def isGloballyUncorrelated(
    unprunedMatrixTable: MatrixValue,
    locallyPrunedTable: TableValue,
    r2Threshold: Double,
    windowSize: Int,
  ): Boolean = {

    val locallyPrunedRDD = getLocallyPrunedRDDWithGT(unprunedMatrixTable, locallyPrunedTable)

    val r2Matrix = LocalLDPruneSuite.correlationMatrixGT(locallyPrunedRDD.map {
      case (_, _, gs) => gs
    }.collect())
    val variantMap = locallyPrunedRDD.zipWithIndex.map { case ((locus, _, _), i) =>
      (i.toInt, locus)
    }.collectAsMap()

    r2Matrix.indices.forall { case (i, j) =>
      val locus1 = variantMap(i)
      val locus2 = variantMap(j)
      val r2 = r2Matrix(i, j)

      locus1 == locus2 ||
      locus1.contig != locus2.contig ||
      (locus1.contig == locus2.contig && math.abs(
        locus1.position - locus2.position
      ) > windowSize) ||
      r2.exists(_ < r2Threshold)
    }
  }

  def isLocallyUncorrelated(
    unprunedMatrixTable: MatrixValue,
    locallyPrunedTable: TableValue,
    r2Threshold: Double,
    windowSize: Int,
  ): Boolean = {

    val locallyPrunedRDD = getLocallyPrunedRDDWithGT(unprunedMatrixTable, locallyPrunedTable)

    val locallyUncorrelated = {
      locallyPrunedRDD.mapPartitions(
        it => {
          // bind function for serialization
          val computeCorrelationMatrix = (gts: Array[Iterable[Annotation]]) =>
            LocalLDPruneSuite.correlationMatrixGT(gts)

          val (it1, it2) = it.duplicate
          val localR2Matrix = computeCorrelationMatrix(it1.map { case (_, _, gs) =>
            gs
          }.toArray)
          val localVariantMap = it2.zipWithIndex.map { case ((locus, _, _), i) =>
            (i, locus)
          }.toMap

          val uncorrelated = localR2Matrix.indices.forall { case (i, j) =>
            val locus1 = localVariantMap(i)
            val locus2 = localVariantMap(j)
            val r2 = localR2Matrix(i, j)

            locus1 == locus2 ||
            locus1.contig != locus2.contig ||
            (locus1.contig == locus2.contig && math.abs(
              locus1.position - locus2.position
            ) > windowSize) ||
            r2.exists(_ < r2Threshold)
          }

          Iterator(uncorrelated)
        },
        preservesPartitioning = true,
      )
    }

    locallyUncorrelated.fold(true)((bool1, bool2) => bool1 && bool2)
  }

  @Test def testBitPackUnpack(): Unit = {
    val calls1 = Array(-1, 0, 1, 2, 1, 1, 0, 0, 0, 0, 2, 2, -1, -1, -1, -1).map(toC2)
    val calls2 = Array(0, 1, 2, 2, 2, 0, -1, -1).map(toC2)
    val calls3 = calls1 ++ Array.ofDim[Int](32 - calls1.length).map(toC2) ++ calls2

    for (calls <- Array(calls1, calls2, calls3))
      assert(LocalLDPruneSuite.fromCalls(calls).forall { bpv =>
        bpv.unpack().map(toC2(_)) sameElements calls
      })
  }

  @Test def testR2(): Unit = {
    val calls = Array(
      Array(1, 0, 0, 0, 0, 0, 0, 0).map(toC2),
      Array(1, 1, 1, 1, 1, 1, 1, 1).map(toC2),
      Array(1, 2, 2, 2, 2, 2, 2, 2).map(toC2),
      Array(1, 0, 0, 0, 1, 1, 1, 1).map(toC2),
      Array(1, 0, 0, 0, 1, 1, 2, 2).map(toC2),
      Array(1, 0, 1, 1, 2, 2, 0, 1).map(toC2),
      Array(1, 0, 1, 0, 2, 2, 1, 1).map(toC2),
    )

    val actualR2 = new MultiArray2(
      7,
      7,
      fs.readLines("src/test/resources/ldprune_corrtest.txt")(_.flatMap(_.map { line =>
        line.trim.split("\t").map(r2 => if (r2 == "NA") None else Some(r2.toDouble))
      }.value).toArray),
    )

    val computedR2 = LocalLDPruneSuite.correlationMatrix(calls)

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
    val bvi1 = LocalLDPruneSuite.fromCalls(input).get
    val bvi2 = LocalLDPruneSuite.fromCalls(input).get

    assert(D_==(LocalLDPrune.computeR2(bvi1, bvi2), 1d))
  }

  object Spec extends Properties("LDPrune") {
    val vectorGen = for {
      nSamples: Int <- Gen.choose(1, 1000)
      v1: Array[BoxedCall] <- Gen.buildableOfN[Array](nSamples, Gen.choose(-1, 2).map(toC2))
      v2: Array[BoxedCall] <- Gen.buildableOfN[Array](nSamples, Gen.choose(-1, 2).map(toC2))
    } yield (nSamples, v1, v2)

    property("bitPacked pack and unpack give same as orig") =
      forAll(vectorGen) { case (_: Int, v1: Array[BoxedCall], _) =>
        val bpv = LocalLDPruneSuite.fromCalls(v1)

        bpv match {
          case Some(x) => LocalLDPruneSuite.fromCalls(x.unpack().map(toC2)).get.gs sameElements x.gs
          case None => true
        }
      }

    property("R2 bitPacked same as BVector") =
      forAll(vectorGen) { case (nSamples: Int, v1: Array[BoxedCall], v2: Array[BoxedCall]) =>
        val bv1 = LocalLDPruneSuite.fromCalls(v1)
        val bv2 = LocalLDPruneSuite.fromCalls(v2)

        val sgs1 =
          LocalLDPruneSuite.normalizedHardCalls(v1).map(math.sqrt(1d / nSamples) * BVector(_))
        val sgs2 =
          LocalLDPruneSuite.normalizedHardCalls(v2).map(math.sqrt(1d / nSamples) * BVector(_))

        (bv1, bv2, sgs1, sgs2) match {
          case (Some(a), Some(b), Some(c: BVector[Double]), Some(d: BVector[Double])) =>
            val rBreeze = c.dot(d): Double
            val r2Breeze = rBreeze * rBreeze
            val r2BitPacked = LocalLDPrune.computeR2(a, b)

            val isSame =
              D_==(r2BitPacked, r2Breeze) && D_>=(r2BitPacked, 0d) && D_<=(r2BitPacked, 1d)
            if (!isSame) {
              println(s"breeze=$r2Breeze bitPacked=$r2BitPacked nSamples=$nSamples")
            }
            isSame
          case _ => true
        }
      }
  }

  @Test def testRandom(): Unit =
    Spec.check()

  @Test def testIsLocallyUncorrelated(): Unit = {
    val locallyPrunedVariantsTable =
      LocalLDPrune(ctx, mt, r2Threshold = 0.2, windowSize = 1000000, maxQueueSize = maxQueueSize)
    assert(isLocallyUncorrelated(mt, locallyPrunedVariantsTable, 0.2, 1000000))
    assert(!isGloballyUncorrelated(mt, locallyPrunedVariantsTable, 0.2, 1000000))
  }
}
