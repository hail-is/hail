package is.hail.methods

import java.util
import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.expr.ir.{LongArrayBuilder, MatrixValue, TableValue}
import is.hail.sparkextras.ContextRDD
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.rvd.RVD
import is.hail.utils._
import is.hail.variant._

import org.apache.spark.rdd.RDD

import BitPackedVector._

object BitPackedVector {
  final val GENOTYPES_PER_PACK: Int = 32
  final val BITS_PER_PACK: Int = 2 * GENOTYPES_PER_PACK
}

private case class BitPackedVectorLoader(fullRowPType: PStruct, callField: String, nSamples: Int) {
  require(nSamples >= 0)
  val PField(_, locusType: PLocus, locusIdx) = fullRowPType.fieldByName("locus")
  val PField(_, allelesType: PArray, allelesIdx) = fullRowPType.fieldByName("alleles")
  val alleleType = allelesType.elementType.asInstanceOf[PString]
  val hcView = new HardCallView(fullRowPType, callField)

  def load(ptr: Long): Option[BitPackedVector] = {
    require(nSamples >= 0)
    hcView.set(ptr)

    val lptr = fullRowPType.loadField(ptr, locusIdx)
    val locus = Locus(locusType.contig(lptr), locusType.position(lptr))

    val aptr = fullRowPType.loadField(ptr, allelesIdx)
    val alen = allelesType.loadLength(aptr)
    val aiter = allelesType.elementIterator(aptr, alen)
    val alleles = new Array[String](alen)
    var i = 0
    while (i < alen) {
      alleles(i) = alleleType.loadString(allelesType.loadElement(aptr, alen, i))
      i += 1
    }

    var nMissing = 0
    var gtSum = 0
    var gtSumSq = 0

    val nPacks = (nSamples - 1) / GENOTYPES_PER_PACK + 1
    val packs = new LongArrayBuilder(nPacks)
    var pack = 0L
    var packOffset = BITS_PER_PACK - 2
    i = 0
    while (i < nSamples) {
      hcView.setGenotype(i)
      val gt = if (hcView.hasGT) Call.nNonRefAlleles(hcView.getGT) else -1
      pack = pack | ((gt & 3).toLong << packOffset)

      if (packOffset == 0) {
        packs.add(pack)
        pack = 0L
        packOffset = BITS_PER_PACK
      }

      packOffset -= 2

      gt match {
        case 1 => gtSum += 1; gtSumSq += 1
        case 2 => gtSum += 2; gtSumSq += 4
        case -1 => nMissing += 1
        case _ =>
      }

      i += 1
    }

    if (packs.size < nPacks) {
      packs.add(pack)
    }

    val nPresent = nSamples - nMissing
    val allHomRef = gtSum == 0
    val allHet = gtSum == nPresent && gtSumSq == nPresent
    val allHomVar = gtSum == 2 * nPresent

    if (allHomRef || allHet || allHomVar || nMissing == nSamples) {
      None
    } else {
      val gtMean = gtSum.toDouble / nPresent
      val gtSumAll = gtSum + nMissing * gtMean
      val gtSumSqAll = gtSumSq + nMissing * gtMean * gtMean
      val gtCenteredLengthRec = 1d / math.sqrt(gtSumSqAll - (gtSumAll * gtSumAll / nSamples))

      Some(BitPackedVector(locus, alleles, packs.result(), nSamples, gtMean, gtCenteredLengthRec))
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

  private def pruneLocal(inputRDD: RDD[BitPackedVector], r2Threshold: Double, windowSize: Int, queueSize: Int): RDD[BitPackedVector] = {
    inputRDD.mapPartitions({ it =>
      val queue = new util.ArrayDeque[BitPackedVector](queueSize)
      it.filter { bpvv =>
        var keepVariant = true
        var done = false
        val qit = queue.descendingIterator()

        while (!done && qit.hasNext) {
          val bpvvPrev = qit.next()
          if (bpvv.locus.contig != bpvvPrev.locus.contig || bpvv.locus.position - bpvvPrev.locus.position > windowSize) {
            done = true
          } else {
            val r2 = computeR2(bpvv, bpvvPrev)
            if (r2 >= r2Threshold) {
              keepVariant = false
              done = true
            }
          }
        }

        if (keepVariant) {
          queue.addLast(bpvv)
          if (queue.size() > queueSize) {
            queue.pop()
          }
        }

        keepVariant
      }
    }, preservesPartitioning = true)
  }

  def apply(ctx: ExecuteContext,
    mt: MatrixValue,
    callField: String = "GT", r2Threshold: Double = 0.2, windowSize: Int = 1000000, maxQueueSize: Int
  ): TableValue = {
    val pruner = LocalLDPrune(callField, r2Threshold, windowSize, maxQueueSize)
    pruner.execute(ctx, mt)
  }
}

case class LocalLDPrune(
  callField: String, r2Threshold: Double, windowSize: Int, maxQueueSize: Int
) extends MatrixToTableFunction {
  require(maxQueueSize > 0, s"Maximum queue size must be positive. Found '$maxQueueSize'.")

  override def typ(childType: MatrixType): TableType = {
    TableType(
      rowType = childType.rowKeyStruct ++ TStruct("mean" -> TFloat64, "centered_length_rec" -> TFloat64),
      key = childType.rowKey, globalType = TStruct.empty)
  }

  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val nSamples = mv.nCols
    val fullRowPType = mv.rvRowPType
    val localCallField = callField

    val tableType = typ(mv.typ)

    val standardizedRDD = mv.rvd
      .mapPartitions{ (ctx, prevPartition) =>
        val bpvLoader = BitPackedVectorLoader(fullRowPType, localCallField, nSamples)
        prevPartition.flatMap { ptr =>
          bpvLoader.load(ptr)
        }
      }

    val rddLP = LocalLDPrune.pruneLocal(standardizedRDD, r2Threshold, windowSize, maxQueueSize)

    val sitesOnly = RVD(tableType.canonicalRVDType, mv.rvd.partitioner, ContextRDD.weaken(rddLP).cmapPartitions({ (ctx, it) =>
      val rvb = new RegionValueBuilder(ctx.r)
      it.map { bpvv =>
        rvb.set(ctx.r)
        rvb.start(tableType.canonicalRowPType)
        rvb.startStruct()
        rvb.addLocus(bpvv.locus.contig, bpvv.locus.position)
        rvb.addAnnotation(TArray(TString), bpvv.alleles)
        rvb.addDouble(bpvv.mean)
        rvb.addDouble(bpvv.centeredLengthRec)
        rvb.endStruct()
        rvb.end()
      }
    }, true))

    TableValue(ctx, tableType, BroadcastRow.empty(ctx), sitesOnly)
  }
}
