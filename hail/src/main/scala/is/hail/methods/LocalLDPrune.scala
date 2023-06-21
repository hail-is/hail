package is.hail.methods

import cats.syntax.all.{toFlatMapOps, toFunctorOps}
import is.hail.expr.ir._
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.expr.ir.lowering.MonadLower
import is.hail.methods.BitPackedVector._
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD

import java.util
import scala.language.higherKinds

object BitPackedVector {
  final val GENOTYPES_PER_PACK: Int = 32
  final val BITS_PER_PACK: Int = 2 * GENOTYPES_PER_PACK
}

class BitPackedVectorBuilder(nSamples: Int) {
  require(nSamples >= 0)
  private var idx = 0
  private var nMissing = 0
  private var gtSum = 0
  private var gtSumSq = 0

  private val nPacks = (nSamples - 1) / GENOTYPES_PER_PACK + 1
  private val packs = new LongArrayBuilder(nPacks)
  private var pack = 0L
  private var packOffset = BITS_PER_PACK - 2

  def reset(): Unit = {
    idx = 0
    nMissing = 0
    gtSum = 0
    gtSumSq = 0
    packs.clear()
    pack = 0L
    packOffset = BITS_PER_PACK - 2
  }

  def addGT(call: Int): Unit = {
    require(idx < nSamples)
    if (!Call.isDiploid(call)) {
      fatal(s"hail LD prune does not support non-diploid calls, found ${Call.toString(call)}")
    }
    val gt = Call.nNonRefAlleles(call)
    pack = pack | ((gt & 3).toLong << packOffset)

    if (packOffset == 0) {
      packs.add(pack)
      pack = 0L
      packOffset = BITS_PER_PACK
    }

    packOffset -= 2

    if (gt == 1) {
      gtSum += 1; gtSumSq += 1
    } else if (gt == 2) {
      gtSum += 2; gtSumSq += 4
    }

    idx += 1
  }

  def addMissing(): Unit = {
    require(idx < nSamples)
    val gt = -1
    pack = pack | ((gt & 3).toLong << packOffset)

    if (packOffset == 0) {
      packs.add(pack)
      pack = 0L
      packOffset = BITS_PER_PACK
    }

    packOffset -= 2
    nMissing += 1
    idx += 1
  }

  def finish(locus: Locus, alleles: Array[String]): BitPackedVector = {
    require(idx == nSamples)

    if (packs.size < nPacks) {
      packs.add(pack)
    }

    val nPresent = nSamples - nMissing
    val allHomRef = gtSum == 0
    val allHet = gtSum == nPresent && gtSumSq == nPresent
    val allHomVar = gtSum == 2 * nPresent

    if (allHomRef || allHet || allHomVar || nMissing == nSamples) {
      null
    } else {
      val gtMean = gtSum.toDouble / nPresent
      val gtSumAll = gtSum + nMissing * gtMean
      val gtSumSqAll = gtSumSq + nMissing * gtMean * gtMean
      val gtCenteredLengthRec = 1d / math.sqrt(gtSumSqAll - (gtSumAll * gtSumAll / nSamples))

      BitPackedVector(locus, alleles, packs.result(), nSamples, gtMean, gtCenteredLengthRec)
    }
  }
}

case class BitPackedVector(locus: Locus, alleles: IndexedSeq[String], gs: Array[Long], nSamples: Int, mean: Double, centeredLengthRec: Double) {
  def nPacks: Int = gs.length

  def getPack(idx: Int): Long = gs(idx)

  // for testing
  private[methods] def unpack(): Array[Int] = {
    val gts = Array.ofDim[Int](nSamples)

    var packIndex = 0
    var i = 0
    val shiftInit = GENOTYPES_PER_PACK * 2 - 2
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

object LocalLDPrune {

  val lookupTable: Array[Byte] = {
    val table = Array.ofDim[Byte](256 * 4)

    (0 until 256).foreach { i =>
      val xi = i & 3
      val xj = (i >> 2) & 3
      val yi = (i >> 4) & 3
      val yj = (i >> 6) & 3

      val res = doubleSampleLookup(xi, yi, xj, yj)

      table(i * 4) = res._1.toByte
      table(i * 4 + 1) = res._2.toByte
      table(i * 4 + 2) = res._3.toByte
      table(i * 4 + 3) = res._4.toByte
    }
    table
  }

  private def doubleSampleLookup(sample1VariantX: Int, sample1VariantY: Int, sample2VariantX: Int,
    sample2VariantY: Int): (Int, Int, Int, Int) = {
    val r1 = singleSampleLookup(sample1VariantX, sample1VariantY)
    val r2 = singleSampleLookup(sample2VariantX, sample2VariantY)
    (r1._1 + r2._1, r1._2 + r2._2, r1._3 + r2._3, r1._4 + r2._4)
  }

  private def singleSampleLookup(xi: Int, yi: Int): (Int, Int, Int, Int) = {
    var xySum = 0
    var XbarCount = 0
    var YbarCount = 0
    var XbarYbarCount = 0

    (xi, yi) match {
      case (3, 3) => XbarYbarCount += 1
      case (3, 1) => XbarCount += 1
      case (3, 2) => XbarCount += 2
      case (1, 3) => YbarCount += 1
      case (2, 3) => YbarCount += 2
      case (2, 2) => xySum += 4
      case (2, 1) => xySum += 2
      case (1, 2) => xySum += 2
      case (1, 1) => xySum += 1
      case _ =>
    }

    (xySum, XbarCount, YbarCount, XbarYbarCount)
  }

  def computeR(x: BitPackedVector, y: BitPackedVector): Double = {
    require(x.nSamples == y.nSamples && x.nPacks == y.nPacks)

    val N = x.nSamples
    val meanX = x.mean
    val meanY = y.mean
    val centeredLengthRecX = x.centeredLengthRec
    val centeredLengthRecY = y.centeredLengthRec

    var XbarYbarCount = 0
    var XbarCount = 0
    var YbarCount = 0
    var xySum = 0

    val nPacks = x.nPacks
    val shiftInit = 2 * (GENOTYPES_PER_PACK - 2)
    var pack = 0
    while (pack < nPacks) {
      val lX = x.getPack(pack)
      val lY = y.getPack(pack)
      var shift = shiftInit

      while (shift >= 0) {
        val b = (((lY >> shift) & 15) << 4 | ((lX >> shift) & 15)).toInt
        xySum += lookupTable(b * 4)
        XbarCount += lookupTable(b * 4 + 1)
        YbarCount += lookupTable(b * 4 + 2)
        XbarYbarCount += lookupTable(b * 4 + 3)
        shift -= 4
      }
      pack += 1
    }

    centeredLengthRecX * centeredLengthRecY *
      ((xySum + XbarCount * meanX + YbarCount * meanY + XbarYbarCount * meanX * meanY) - N * meanX * meanY)
  }

  def computeR2(x: BitPackedVector, y: BitPackedVector): Double = {
    val r = computeR(x, y)
    val r2 = r * r
    assert(D_>=(r2, 0d) && D_<=(r2, 1d), s"R2 must lie in [0,1]. Found $r2.")
    r2
  }

  def pruneLocal(queue: util.ArrayDeque[BitPackedVector], bpv: BitPackedVector, r2Threshold: Double, windowSize: Int, queueSize: Int): Boolean = {
    var keepVariant = true
    var done = false
    val qit = queue.descendingIterator()

    while (!done && qit.hasNext) {
      val bpvPrev = qit.next()
      if (bpv.locus.contig != bpvPrev.locus.contig || bpv.locus.position - bpvPrev.locus.position > windowSize) {
        done = true
      } else {
        val r2 = computeR2(bpv, bpvPrev)
        if (r2 >= r2Threshold) {
          keepVariant = false
          done = true
        }
      }
    }

    if (keepVariant) {
      queue.addLast(bpv)
      if (queue.size() > queueSize) {
        queue.pop()
      }
    }

    keepVariant
  }

  private def pruneLocal(inputRDD: RDD[BitPackedVector], r2Threshold: Double, windowSize: Int, queueSize: Int): RDD[BitPackedVector] = {
    inputRDD.mapPartitions({ it =>
      val queue = new util.ArrayDeque[BitPackedVector](queueSize)
      it.filter { bpvv =>
        pruneLocal(queue, bpvv, r2Threshold, windowSize, queueSize)
      }
    }, preservesPartitioning = true)
  }

  def apply[M[_]: MonadLower](mt: MatrixValue,
                              callField: String = "GT",
                              r2Threshold: Double = 0.2,
                              windowSize: Int = 1000000,
                              maxQueueSize: Int
                             ): M[TableValue] =
    LocalLDPrune(callField, r2Threshold, windowSize, maxQueueSize).execute(mt)
}

case class LocalLDPrune(
  callField: String, r2Threshold: Double, windowSize: Int, maxQueueSize: Int
) extends MatrixToTableFunction {
  require(maxQueueSize > 0, s"Maximum queue size must be positive. Found '$maxQueueSize'.")

  override def typ(childType: MatrixType): TableType =
    TableType(
      rowType = childType.rowKeyStruct ++ TStruct("mean" -> TFloat64, "centered_length_rec" -> TFloat64),
      key = childType.rowKey, globalType = TStruct.empty)

  def preservesPartitionCounts: Boolean = false

  def makeStream(stream: IR, entriesFieldName: String, nCols: IR): StreamLocalLDPrune = {
    val newRow = mapIR(stream) { row =>
      val entries = ToStream(GetField(row, entriesFieldName))
      val genotypes = ToArray(mapIR(entries)(ent => GetField(ent, callField)))
      val locus = GetField(row, "locus")
      val alleles = GetField(row, "alleles")
      makestruct("locus" -> locus, "alleles" -> alleles, "genotypes" -> genotypes)
    }
    StreamLocalLDPrune(newRow, r2Threshold, windowSize, maxQueueSize, nCols)
  }

  def execute[M[_]](mv: MatrixValue)(implicit M: MonadLower[M]): M[TableValue] =
    for {
      ts <- TableValueIntermediate(mv.toTableValue).asTableStage
      imm = ts.mapPartition(Some(typ(mv.typ).key)) { rows =>
        makeStream(rows, MatrixType.entriesIdentifier, mv.nCols)
      }.mapGlobals(_ => makestruct())
      tv <- TableStageIntermediate(imm).asTableValue
    } yield tv
}
