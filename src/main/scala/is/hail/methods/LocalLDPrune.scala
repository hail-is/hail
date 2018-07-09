package is.hail.methods

import java.util

import is.hail.annotations._
import is.hail.expr.types._
import org.apache.spark.storage.StorageLevel
import is.hail.rvd.{OrderedRVD, OrderedRVDType}
import is.hail.table.Table
import is.hail.variant._
import is.hail.utils._

object BitPackedVectorView {
  val bpvElementSize = TInt64Required.byteSize

  def rvRowType(locusType: Type, allelesType: Type): TStruct = TStruct("locus" -> locusType, "alleles" -> allelesType,
    "bpv" -> TArray(TInt64Required), "nSamples" -> TInt32Required, "mean" -> TFloat64Required, "centered_length_rec" -> TFloat64Required)
}

class BitPackedVectorView(rvRowType: TStruct) {
  val vView = new RegionValueVariant(rvRowType)

  // All types are required!
  private var m: Region = _
  private var bpvOffset: Long = _
  private var bpvLength: Int = _
  private var bpvElementOffset: Long = _
  private var nSamplesOffset: Long = _
  private var meanOffset: Long = _
  private var centeredLengthRecOffset: Long = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    bpvOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("bpv"))
    bpvLength = TArray(TInt64Required).loadLength(m, bpvOffset)
    bpvElementOffset = TArray(TInt64Required).elementOffset(bpvOffset, bpvLength, 0)
    nSamplesOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("nSamples"))
    meanOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("mean"))
    centeredLengthRecOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("centered_length_rec"))

    vView.setRegion(m, offset)
  }

  def setRegion(rv: RegionValue): Unit = setRegion(rv.region, rv.offset)

  def getContig: String = vView.contig()

  def getStart: Int = vView.position()

  def getPack(idx: Int): Long = {
    if (idx < 0 || idx >= bpvLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    val packOffset = bpvElementOffset + idx * BitPackedVectorView.bpvElementSize
    m.loadLong(packOffset)
  }

  def getNPacks: Int = bpvLength

  def getNSamples: Int = m.loadInt(nSamplesOffset)

  def getMean: Double = m.loadDouble(meanOffset)

  def getCenteredLengthRec: Double = m.loadDouble(centeredLengthRecOffset)
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

  private def pruneLocal(inputRDD: OrderedRVD, r2Threshold: Double, windowSize: Int, queueSize: Option[Int]): OrderedRVD = {
    val localRowType = inputRDD.typ.rowType

    inputRDD.mapPartitionsPreservesPartitioning(inputRDD.typ, { (ctx, it) =>
      val queue = new util.ArrayDeque[RegionValue](queueSize.getOrElse(16))

      val bpvv = new BitPackedVectorView(localRowType)
      val bpvvPrev = new BitPackedVectorView(localRowType)
      val rvb = new RegionValueBuilder()

      it.filter { rv =>
        bpvv.setRegion(rv)

        var keepVariant = true
        var done = false
        val qit = queue.descendingIterator()

        while (!done && qit.hasNext) {
          bpvvPrev.setRegion(qit.next())
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
          val r = ctx.freshRegion
          rvb.set(r)
          rvb.start(localRowType)
          rvb.addRegionValue(localRowType, rv)
          queue.addLast(RegionValue(rvb.region, rvb.end()))
          queueSize.foreach { qs =>
            if (queue.size() > qs) {
              ctx.closeChild(queue.pop().region)
            }
          }
        }

        keepVariant
      }
    })
  }

  def apply(mt: MatrixTable, callField: String = "GT", r2Threshold: Double = 0.2, windowSize: Int = 1000000, maxQueueSize: Int): Table = {
    if (maxQueueSize < 1)
      fatal(s"Maximum queue size must be positive. Found `$maxQueueSize'.")

    val nSamples = mt.numCols
    
    val fullRowType = mt.rvRowType

    val locusIndex = mt.rvRowType.fieldIdx("locus")
    val allelesIndex = mt.rvRowType.fieldIdx("alleles")

    val bpvType = BitPackedVectorView.rvRowType(mt.rvRowType.types(locusIndex), mt.rvRowType.types(allelesIndex))

    val typ = mt.rvd.typ

    val standardizedRDD = mt.rvd
      .mapPartitionsPreservesPartitioning(new OrderedRVDType(typ.partitionKey, typ.key, bpvType))({ it =>
        val hcView = new HardCallView(fullRowType, callField)
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val newRV = RegionValue(region)

        it.flatMap { rv =>
          hcView.setRegion(rv)
          region.clear()
          rvb.set(region)
          rvb.start(bpvType)
          rvb.startStruct()
          rvb.addFields(fullRowType, rv, Array(locusIndex, allelesIndex))

          val keep = addBitPackedVector(rvb, hcView, nSamples)

          if (keep) {
            rvb.endStruct()
            newRV.setOffset(rvb.end())
            Some(newRV)
          }
          else
            None
        }
      })
    
    val rvdLP = pruneLocal(standardizedRDD, r2Threshold, windowSize, Some(maxQueueSize))

    val tableType = TableType(
      rowType = mt.rowKeyStruct ++ TStruct("mean" -> TFloat64Required, "centered_length_rec" -> TFloat64Required),
      key = Some(mt.rowKey), globalType = TStruct.empty())

    val sitesOnly = rvdLP.mapPartitionsPreservesPartitioning(
      new OrderedRVDType(typ.partitionKey, typ.key, tableType.rowType))({
      it =>
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val newRV = RegionValue(region)

        it.map { rv =>
          region.clear()
          rvb.set(region)
          rvb.start(tableType.rowType)
          rvb.startStruct()
          rvb.addFields(bpvType, rv, Array("locus", "alleles", "mean", "centered_length_rec")
            .map(field => bpvType.fieldIdx(field)))
          rvb.endStruct()
          newRV.setOffset(rvb.end())
          newRV
        }
    })

    new Table(hc = mt.hc, crdd = sitesOnly.crdd, signature = tableType.rowType, key = tableType.key)
  }
}

