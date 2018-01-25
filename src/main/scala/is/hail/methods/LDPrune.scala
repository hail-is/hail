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
    "bpv" -> TArray(TInt64Required), "nSamples" -> TInt32Required, "mean" -> TFloat64Required, "sd" -> TFloat64Required)
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
  private var sdOffset: Long = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    bpvOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("bpv"))
    bpvLength = TArray(TInt64Required).loadLength(m, bpvOffset)
    bpvElementOffset = TArray(TInt64Required).elementOffset(bpvOffset, bpvLength, 0)
    nSamplesOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("nSamples"))
    meanOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("mean"))
    sdOffset = rvRowType.loadField(m, offset, rvRowType.fieldIdx("sd"))

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

  def getStdDevRecip: Double = m.loadDouble(sdOffset)
}

object LDPrune {
  val variantByteOverhead = 50
  val fractionMemoryToUse = 0.25
  val genotypesPerPack = 32
  val nPartitionsPerCore = 3

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
      val gt = if (hcView.hasGT) Call.unphasedDiploidGtIndex(hcView.getGT) else -1

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
      val gtMeanAll = (gtSum + nMissing * gtMean) / nSamples
      val gtMeanSqAll = (gtSumSq + nMissing * gtMean * gtMean) / nSamples
      val gtStdDevRec = 1d / math.sqrt((gtMeanSqAll - gtMeanAll * gtMeanAll) * nSamples)

      rvb.addInt(nSamples)
      rvb.addDouble(gtMean)
      rvb.addDouble(gtStdDevRec)
      true
    }
  }

  def computeR(x: BitPackedVectorView, y: BitPackedVectorView): Double = {
    require(x.getNSamples == y.getNSamples && x.getNPacks == y.getNPacks)

    val N = x.getNSamples
    val meanX = x.getMean
    val meanY = y.getMean
    val stdDevRecX = x.getStdDevRecip
    val stdDevRecY = y.getStdDevRecip

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

    stdDevRecX * stdDevRecY * ((xySum + XbarCount * meanX + YbarCount * meanY + XbarYbarCount * meanX * meanY) - N * meanX * meanY)
  }

  def computeR2(x: BitPackedVectorView, y: BitPackedVectorView): Double = {
    val r = computeR(x, y)
    val r2 = r * r
    assert(D_>=(r2, 0d) && D_<=(r2, 1d), s"R2 must lie in [0,1]. Found $r2.")
    r2
  }

  private def pruneLocal(inputRDD: OrderedRVD, r2Threshold: Double, windowSize: Int, queueSize: Option[Int]): OrderedRVD = {
    val localRowType = inputRDD.typ.rowType

    inputRDD.mapPartitionsPreservesPartitioning(inputRDD.typ) { it =>
      val queue = new util.ArrayDeque[RegionValue](queueSize.getOrElse(16))

      val bpvv = new BitPackedVectorView(localRowType)
      val bpvvPrev = new BitPackedVectorView(localRowType)

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
          queue.addLast(rv.copy())
          queueSize.foreach { qs =>
            if (queue.size() > qs) {
              queue.pop()
            }
          }
        }

        keepVariant
      }
    }
  }

  private def pruneLocalTimed(inputRDD: OrderedRVD, r2Threshold: Double, windowSize: Int, maxQueueSize: Option[Int]):
  ((OrderedRVD, Long, Int), Long) = {
    time({
      val prunedRDD = pruneLocal(inputRDD, r2Threshold, windowSize, maxQueueSize).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      (prunedRDD, nVariantsKept, nPartitions)
    })
  }


  def estimateMemoryRequirements(nVariants: Long, nSamples: Int, nCores: Int, memoryPerCore: Long) = {
    val nBytesPerVariant = math.ceil(8 * nSamples.toDouble / genotypesPerPack).toLong + variantByteOverhead
    val memoryAvailPerCore = memoryPerCore * fractionMemoryToUse

    val maxQueueSize = math.max(1, math.ceil(memoryAvailPerCore / nBytesPerVariant).toInt)

    val nPartitionsMinimum = math.max(1, math.ceil(nVariants.toDouble / maxQueueSize).toInt)
    val nPartitionsOptimal = nCores * nPartitionsPerCore

    val nPartitions =
      if (nPartitionsOptimal < nPartitionsMinimum)
        nPartitionsMinimum
      else
        nPartitionsOptimal

    assert(maxQueueSize > 0 && nPartitions > 0)

    (maxQueueSize, nPartitions)
  }

  def apply(mt: MatrixTable, nCores: Int, r2Threshold: Double = 0.2, windowSize: Int = 1000000,
    memoryPerCoreMB: Long = 256): Table = {

    mt.requireRowKeyVariant("ld_prune")

    val memoryPerCore = memoryPerCoreMB * 1024L * 1024L

    if (nCores <= 0)
      fatal(s"Number of cores must be positive.")

    if (r2Threshold < 0 || r2Threshold > 1)
      fatal(s"R^2 threshold must be in the range [0,1]. Found `$r2Threshold'.")

    if (windowSize < 0)
      fatal(s"Window size must be greater than or equal to 0. Found `$windowSize'.")

    val (nVariantsInitial, nPartitionsInitial, nSamples) = (mt.countRows(), mt.nPartitions, mt.numCols)

    val minMemoryPerCore = math.ceil((1d / fractionMemoryToUse) * 8 * nSamples + variantByteOverhead)
    val (maxQueueSize, _) = estimateMemoryRequirements(nVariantsInitial, nSamples, nCores, memoryPerCore)

    info(s"Running LD prune with nSamples=$nSamples, nVariants=$nVariantsInitial, nPartitions=$nPartitionsInitial, and maxQueueSize=$maxQueueSize.")

    if (memoryPerCore < minMemoryPerCore)
      fatal(s"`memory_per_core' must be greater than ${ minMemoryPerCore / (1024 * 1024) }MB.")

    val fullRowType = mt.rvRowType

    val locusIndex = mt.rowType.fieldIdx("locus")
    val allelesIndex = mt.rowType.fieldIdx("alleles")

    val bpvType = BitPackedVectorView.rvRowType(mt.rowKeyTypes(locusIndex), mt.rowKeyTypes(allelesIndex))

    val typ = mt.rvd.typ

    val standardizedRDD = mt.rvd
      .mapPartitionsPreservesPartitioning(new OrderedRVDType(typ.partitionKey, typ.key, bpvType))({ it =>
        val hcView = HardCallView(fullRowType)
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

    val ((rddLP1, nVariantsLP1, nPartitionsLP1), durationLP1) = pruneLocalTimed(
      standardizedRDD, r2Threshold, windowSize, Some(maxQueueSize))
    info(s"LD prune step 1 of 3: nVariantsKept=$nVariantsLP1, nPartitions=$nPartitionsLP1, time=${ formatTime(durationLP1) }")

    val (_, nPartitionsRequired) = estimateMemoryRequirements(nVariantsLP1, nSamples, nCores, memoryPerCore)
    val repartRDD = rddLP1.coalesce(nPartitionsRequired, shuffle = true).persist(StorageLevel.MEMORY_AND_DISK)
    repartRDD.count()
    rddLP1.unpersist()
    val ((rddLP2, nVariantsLP2, nPartitionsLP2), durationLP2) = pruneLocalTimed(repartRDD, r2Threshold, windowSize, None)
    repartRDD.unpersist()
    info(s"LD prune step 2 of 3: nVariantsKept=$nVariantsLP2, nPartitions=$nPartitionsLP2, time=${ formatTime(durationLP2) }")
    
    val tableType = TableType(
      rowType = mt.rowKeyStruct ++ TStruct("mean" -> TFloat64Required, "sd_reciprocal" -> TFloat64Required),
      key = mt.rowKey, globalType = TStruct.empty())

    val sitesOnlyRDD = rddLP2.mapPartitionsPreservesPartitioning(
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
          rvb.addFields(bpvType, rv, Array("locus", "alleles", "mean", "sd").map(field => bpvType.fieldIdx(field)))
          rvb.endStruct()
          newRV.setOffset(rvb.end())
          newRV
        }
    })

    new Table(hc = mt.hc, rdd = sitesOnlyRDD.rdd, signature = tableType.rowType, key = tableType.key)
  }
}

