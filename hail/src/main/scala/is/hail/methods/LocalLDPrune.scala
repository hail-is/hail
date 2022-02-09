package is.hail.methods

import java.util
import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.rvd.RVD
import is.hail.utils._
import is.hail.variant._

object BitPackedVectorView {
  val bpvElementSize: Long = PInt64Required.byteSize

  def rvRowPType(locusType: PType, allelesType: PType): PStruct =
    PCanonicalStruct(true,
      "locus" -> locusType,
      "alleles" -> allelesType,
      "bpv" -> PCanonicalArray(PInt64Required),
      "nSamples" -> PInt32Required,
      "mean" -> PFloat64(),
      "centered_length_rec" -> PFloat64())
}

class BitPackedVectorView(rvRowType: PStruct) {
  val vView = new RegionValueVariant(rvRowType)

  // All types are required!
  private var bpvOffset: Long = _
  private var bpvLength: Int = _
  private var bpvElementOffset: Long = _
  private var nSamplesOffset: Long = _
  private var meanOffset: Long = _
  private var centeredLengthRecOffset: Long = _
  private val bpvPType = PCanonicalArray(PInt64Required)

  def set(offset: Long) {
    bpvOffset = rvRowType.loadField(offset, rvRowType.fieldIdx("bpv"))
    bpvLength = bpvPType.loadLength(bpvOffset)
    bpvElementOffset = bpvPType.elementOffset(bpvOffset, bpvLength, 0)
    nSamplesOffset = rvRowType.loadField(offset, rvRowType.fieldIdx("nSamples"))
    meanOffset = rvRowType.loadField(offset, rvRowType.fieldIdx("mean"))
    centeredLengthRecOffset = rvRowType.loadField(offset, rvRowType.fieldIdx("centered_length_rec"))

    vView.set(offset)
  }

  def getContig: String = vView.contig()

  def getStart: Int = vView.position()

  def getPack(idx: Int): Long = {
    if (idx < 0 || idx >= bpvLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    val packOffset = bpvElementOffset + idx * BitPackedVectorView.bpvElementSize
    Region.loadLong(packOffset)
  }

  def getNPacks: Int = bpvLength

  def getNSamples: Int = Region.loadInt(nSamplesOffset)

  def getMean: Double = Region.loadDouble(meanOffset)

  def getCenteredLengthRec: Double = Region.loadDouble(centeredLengthRecOffset)
}

object LocalLDPrune {
  val genotypesPerPack = 32

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

  def addBitPackedVector(rvb: RegionValueBuilder, hcView: HardCallView, nSamples: Int): Boolean = {
    require(nSamples >= 0)
    val nBitsPerPack = 2 * genotypesPerPack
    val nPacks = (nSamples - 1) / genotypesPerPack + 1

    rvb.startArray(nPacks)

    var nMissing = 0
    var gtSum = 0
    var gtSumSq = 0

    var pack = 0L
    var packOffset = nBitsPerPack - 2
    var packIndex = 0
    var i = 0
    while (i < nSamples) {
      hcView.setGenotype(i)
      val gt = if (hcView.hasGT) Call.nNonRefAlleles(hcView.getGT) else -1

      pack = pack | ((gt & 3).toLong << packOffset)

      if (packOffset == 0) {
        rvb.addLong(pack)
        packIndex += 1
        pack = 0L
        packOffset = nBitsPerPack
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

    if (packIndex < nPacks)
      rvb.addLong(pack)

    rvb.endArray()

    val nPresent = nSamples - nMissing
    val allHomRef = gtSum == 0
    val allHet = gtSum == nPresent && gtSumSq == nPresent
    val allHomVar = gtSum == 2 * nPresent

    if (allHomRef || allHet || allHomVar || nMissing == nSamples) {
      rvb.clear()
      false
    } else {
      val gtMean = gtSum.toDouble / nPresent
      val gtSumAll = gtSum + nMissing * gtMean
      val gtSumSqAll = gtSumSq + nMissing * gtMean * gtMean
      val gtCenteredLengthRec = 1d / math.sqrt(gtSumSqAll - (gtSumAll * gtSumAll / nSamples))

      rvb.addInt(nSamples)
      rvb.addDouble(gtMean)
      rvb.addDouble(gtCenteredLengthRec)
      true
    }
  }

  def computeR(x: BitPackedVectorView, y: BitPackedVectorView): Double = {
    require(x.getNSamples == y.getNSamples && x.getNPacks == y.getNPacks)

    val N = x.getNSamples
    val meanX = x.getMean
    val meanY = y.getMean
    val centeredLengthRecX = x.getCenteredLengthRec
    val centeredLengthRecY = y.getCenteredLengthRec

    var XbarYbarCount = 0
    var XbarCount = 0
    var YbarCount = 0
    var xySum = 0

    val nPacks = x.getNPacks
    val shiftInit = 2 * (genotypesPerPack - 2)
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

  def computeR2(x: BitPackedVectorView, y: BitPackedVectorView): Double = {
    val r = computeR(x, y)
    val r2 = r * r
    assert(D_>=(r2, 0d) && D_<=(r2, 1d), s"R2 must lie in [0,1]. Found $r2.")
    r2
  }

  private def pruneLocal(inputRDD: RVD, r2Threshold: Double, windowSize: Int, queueSize: Option[Int]): RVD = {
    val localRowType = inputRDD.typ.rowType

    inputRDD.mapPartitionsWithContext(inputRDD.typ) { (ctx, prevPartition) =>
      val queue = new util.ArrayDeque[RegionValue](queueSize.getOrElse(16))

      val producerContext = ctx.freshContext()

      val bpvv = new BitPackedVectorView(localRowType)
      val bpvvPrev = new BitPackedVectorView(localRowType)
      val rvb = new RegionValueBuilder()

      prevPartition(producerContext).filter { ptr =>
        bpvv.set(ptr)

        var keepVariant = true
        var done = false
        val qit = queue.descendingIterator()

        while (!done && qit.hasNext) {
          bpvvPrev.set(qit.next().offset)
          if (bpvv.getContig != bpvvPrev.getContig || bpvv.getStart - bpvvPrev.getStart > windowSize)
            done = true
          else {
            val r2 = computeR2(bpvv, bpvvPrev)
            if (r2 >= r2Threshold) {
              keepVariant = false
              done = true
            }
          }
        }

        if (keepVariant) {
          val r = ctx.freshRegion()
          rvb.set(r)
          rvb.start(localRowType)
          rvb.addRegionValue(localRowType, producerContext.region, ptr)
          queue.addLast(RegionValue(rvb.region, rvb.end()))
          queueSize.foreach { qs =>
            if (queue.size() > qs) {
              queue.pop().region.invalidate()
            }
          }
          ctx.r.addReferenceTo(r)
        }

        producerContext.region.clear()

        keepVariant
      }
    }
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

  override def typ(childType: MatrixType): TableType = {
    TableType(
      rowType = childType.rowKeyStruct ++ TStruct("mean" -> TFloat64, "centered_length_rec" -> TFloat64),
      key = childType.rowKey, globalType = TStruct.empty)
  }

  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    if (maxQueueSize < 1)
      fatal(s"Maximum queue size must be positive. Found '$maxQueueSize'.")

    val nSamples = mv.nCols

    val fullRowPType = mv.rvRowPType

    val locusIndex = fullRowPType.fieldIdx("locus")
    val allelesIndex = fullRowPType.fieldIdx("alleles")

    val bpvType = BitPackedVectorView.rvRowPType(fullRowPType.types(locusIndex), fullRowPType.types(allelesIndex))

    val tableType = typ(mv.typ)

    val standardizedRDD = mv.rvd
      .mapPartitionsWithContext(mv.rvd.typ.copy(rowType = bpvType)){ (ctx, prevPartition) =>

        val producerContext = ctx.freshContext()

        val hcView = new HardCallView(fullRowPType, callField)
        val region = producerContext.region
        val rvb = new RegionValueBuilder(region)

        prevPartition(producerContext).flatMap { ptr =>
          hcView.set(ptr)
          rvb.start(bpvType)
          rvb.startStruct()
          rvb.addFields(fullRowPType, ctx.r, ptr, Array(locusIndex, allelesIndex))

          val keep = LocalLDPrune.addBitPackedVector(rvb, hcView, nSamples)

          if (keep) {
            rvb.endStruct()

            ctx.region.addReferenceTo(region)
            region.clear()

            Some(rvb.end())
          }
          else {
            region.clear()
            None
          }
        }
      }

    val rvdLP = LocalLDPrune.pruneLocal(standardizedRDD, r2Threshold, windowSize, Some(maxQueueSize))

    val fieldIndicesToAdd = Array("locus", "alleles", "mean", "centered_length_rec")
      .map(field => bpvType.fieldIdx(field))
    val sitesOnly = rvdLP.mapPartitions(
      tableType.canonicalRVDType
    )({ (ctx, it) =>
      val rvb = new RegionValueBuilder(ctx.r)

      it.map { ptr =>
        rvb.set(ctx.r)
        rvb.start(tableType.canonicalRowPType)
        rvb.startStruct()
        rvb.addFields(bpvType, ctx.r, ptr, fieldIndicesToAdd)
        rvb.endStruct()
        rvb.end()
      }
    })

    TableValue(ctx, tableType, BroadcastRow.empty(ctx), sitesOnly)
  }
}

