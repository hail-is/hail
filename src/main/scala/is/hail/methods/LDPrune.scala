package is.hail.methods

import java.util

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.sparkextras.GeneralRDD
import org.apache.spark.storage.StorageLevel
import is.hail.sparkextras._
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVD, UnpartitionedRVD}
import is.hail.variant._
import is.hail.utils._
import org.apache.spark.sql.Row

object BitPackedVectorView {
  val bpvEltSize = TInt64Required.byteSize

  def rvRowType(locusType: Type, allelesType: Type): TStruct = TStruct("locus" -> locusType, "alleles" -> allelesType,
    "bpv" -> TArray(TInt64Required), "nSamples" -> TInt32Required, "mean" -> TFloat64Required, "sd" -> TFloat64Required)
}

class BitPackedVectorView(rvRowType: TStruct) {
  val vView = new RegionValueVariant(rvRowType)

  // All types are required!
  private val bpvIndex = 2
  private val nSamplesIndex = 3
  private val meanIndex = 4
  private val sdIndex = 5

  private var m: Region = _
  private var vOffset: Long = _
  private var bpvOffset: Long = _
  private var bpvLength: Int = _
  private var bpvEltOffset: Long = _
  private var nSamplesOffset: Long = _
  private var meanOffset: Long = _
  private var sdOffset: Long = _

  def setRegion(mb: Region, offset: Long) {
    this.m = mb
    bpvOffset = rvRowType.loadField(m, offset, bpvIndex)
    bpvLength = TArray(TInt64Required).loadLength(m, bpvOffset)
    bpvEltOffset = TArray(TInt64Required).elementOffset(bpvOffset, bpvLength, 0)
    nSamplesOffset = rvRowType.loadField(m, offset, nSamplesIndex)
    meanOffset = rvRowType.loadField(m, offset, meanIndex)
    sdOffset = rvRowType.loadField(m, offset, sdIndex)

    vView.setRegion(m, vOffset)
  }

  def setRegion(rv: RegionValue): Unit = setRegion(rv.region, rv.offset)

  def getVariant: Variant = Variant.fromRegionValue(m, vOffset)

  def getContig: String = vView.contig()

  def getStart: Int = vView.position()

  def getPack(idx: Int): Long = {
    if (idx < 0 || idx >= bpvLength)
      throw new ArrayIndexOutOfBoundsException(idx)
    val packOffset = bpvEltOffset + idx * BitPackedVectorView.bpvEltSize
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

  case class GlobalPruneIntermediate(rvd: RVD, rvRowType: TStruct, index: Int, persist: Boolean)

  val table: Array[Byte] = {
    val t = Array.ofDim[Byte](256 * 4)

    (0 until 256).foreach { i =>
      val xi = i & 3
      val xj = (i >> 2) & 3
      val yi = (i >> 4) & 3
      val yj = (i >> 6) & 3

      val res = findTableValue(xi, yi, xj, yj)

      t(i * 4) = res._1.toByte
      t(i * 4 + 1) = res._2.toByte
      t(i * 4 + 2) = res._3.toByte
      t(i * 4 + 3) = res._4.toByte
    }
    t
  }

  private def findTableValue(a: Int, b: Int, c: Int, d: Int): (Int, Int, Int, Int) = {
    val r1 = findTableValue(a, b)
    val r2 = findTableValue(c, d)
    (r1._1 + r2._1, r1._2 + r2._2, r1._3 + r2._3, r1._4 + r2._4)
  }

  private def findTableValue(xi: Int, yi: Int): (Int, Int, Int, Int) = {
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

      if (gt == 1) {
        gtSum += 1
        gtSumSq += 1
      } else if (gt == 2) {
        gtSum += 2
        gtSumSq += 4
      } else if (gt == -1) {
        nMissing += 1
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
        xySum += table(b * 4)
        XbarCount += table(b * 4 + 1)
        YbarCount += table(b * 4 + 2)
        XbarYbarCount += table(b * 4 + 3)
        shift -= 4
      }
      pack += 1
    }

    val r = stdDevRecX * stdDevRecY * ((xySum + XbarCount * meanX + YbarCount * meanY + XbarYbarCount * meanX * meanY) - N * meanX * meanY)
    r
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
      val queue = queueSize match {
        case Some(qs) => new util.ArrayDeque[RegionValue](qs)
        case None => new util.ArrayDeque[RegionValue]
      }

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

  private def pruneGlobal(inputRDD: OrderedRVD,
    r2Threshold: Double, windowSize: Int): (OrderedRVD, Long) = {
    val sc = inputRDD.sparkContext

    require(inputRDD.storageLevel == StorageLevel.MEMORY_AND_DISK)

    val partitioner = inputRDD.partitioner
    val rangeBounds = partitioner.rangeBounds.map(a => a.asInstanceOf[Interval]).toArray
    val nPartitions = inputRDD.partitions.length

    val localRowType = inputRDD.typ.rowType

    def computeDependencies(partitionId: Int): Array[Int] = {
      val startLocus = rangeBounds(partitionId).start.asInstanceOf[UnsafeRow].getAs[Locus](0)
      val minLocus = Locus(startLocus.contig, math.max(startLocus.position - windowSize, 0))

      // FIXME: -1 if not in rangeBounds. currently implies minLocus < rvd min,
      // which means filter is fine, but needs to change once rangebounds can have
      // gaps.
      val minPart = partitioner.getPartitionPK(Annotation(minLocus))
      Array.range(minPart, partitionId + 1).reverse
    }

    def pruneF = (x: Array[Iterator[RegionValue]]) => {
      val nPartitions = x.length
      val targetIterator = x(0)
      val prevPartitions = x.drop(1).reverse

      val bpvv = new BitPackedVectorView(localRowType)
      val bpvvPrev = new BitPackedVectorView(localRowType)

      if (nPartitions == 1)
        targetIterator
      else {
        var targetData = targetIterator.map(_.copy()).toArray

        prevPartitions.foreach { it =>
          it.foreach { prevRV =>
            bpvvPrev.setRegion(prevRV)
            targetData = targetData.filter { rv =>
              bpvv.setRegion(rv)
              if (bpvv.getContig != bpvvPrev.getContig || math.abs(bpvv.getStart - bpvvPrev.getStart) > windowSize)
                true
              else {
                computeR2(bpvv, bpvvPrev) < r2Threshold
              }
            }
          }
        }
        targetData.iterator
      }
    }

    val contigStartPartitions = Array.range(0, nPartitions).filter { i =>
        i == 0 || rangeBounds(i-1).end.asInstanceOf[UnsafeRow].getAs[Locus](0).contig != rangeBounds(i).end.asInstanceOf[UnsafeRow].getAs[Locus](0).contig
      }

    val pruneIntermediates = Array.fill[GlobalPruneIntermediate](nPartitions)(null)

    def generalRDDInputs(partitionIndex: Int): (Array[RVD], Array[(Int, Int)]) = {
      val (rvds, inputs) = computeDependencies(partitionIndex).zipWithIndex.map { case (depIndex, i) =>
        if (depIndex == partitionIndex || contigStartPartitions.contains(depIndex))
          (inputRDD, (i, depIndex))
        else {
          val gpi = pruneIntermediates(depIndex)
          pruneIntermediates(depIndex) = gpi.copy(rvd = gpi.rvd.persist(StorageLevel.MEMORY_AND_DISK), persist = true)
          (pruneIntermediates(depIndex).rvd, (i, gpi.index))
        }
      }.unzip
      (rvds.toArray, inputs.toArray)
    }

    for (i <- 0 until nPartitions) {
      val (rvds, inputs) = generalRDDInputs(i)
      pruneIntermediates(i) = GlobalPruneIntermediate(
        rvd = new UnpartitionedRVD(inputRDD.typ.rowType, new GeneralRDD(sc, rvds.map(_.rdd), Array((inputs, pruneF)))),
        rvRowType = localRowType,
        index = 0,
        persist = false) // creating single partition RDDs with partition index = 0
    }

    val prunedRDD = OrderedRVD(inputRDD.typ,
      inputRDD.partitioner,
      new GeneralRDD[RegionValue](sc, pruneIntermediates.map(_.rvd.rdd),
        pruneIntermediates.zipWithIndex.map { case (gpi, i) =>
          (Array((i, gpi.index)), pruneF)
        }))
      .persist(StorageLevel.MEMORY_AND_DISK)

    val nVariantsKept = prunedRDD.count()

    pruneIntermediates.foreach { gpi =>
      if (gpi.persist)
        gpi.rvd.unpersist()
    }
    inputRDD.unpersist()

    (prunedRDD, nVariantsKept)
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

  def apply(vsm: MatrixTable, nCores: Int,
    r2Threshold: Double = 0.2, windowSize: Int = 1000000, memoryPerCoreMB: Long = 256): MatrixTable = {
    vsm.requireRowKeyVariant("ld_prune")

    // in bytes
    val memoryPerCore = memoryPerCoreMB * 1024L * 1024L

    if (nCores <= 0)
      fatal(s"Number of cores must be positive.")

    if (r2Threshold < 0 || r2Threshold > 1)
      fatal(s"R^2 threshold must be in the range [0,1]. Found `$r2Threshold'.")

    if (windowSize < 0)
      fatal(s"Window size must be greater than or equal to 0. Found `$windowSize'.")

    val nVariantsInitial = vsm.countVariants()
    val nPartitionsInitial = vsm.nPartitions
    val nSamples = vsm.numCols

    val minMemoryPerCore = math.ceil((1d / fractionMemoryToUse) * 8 * nSamples + variantByteOverhead)
    val (maxQueueSize, _) = estimateMemoryRequirements(nVariantsInitial, nSamples, nCores, memoryPerCore)

    info(s"Running LD prune with nSamples=$nSamples, nVariants=$nVariantsInitial, nPartitions=$nPartitionsInitial, and maxQueueSize=$maxQueueSize.")

    if (memoryPerCore < minMemoryPerCore)
      fatal(s"`memory_per_core' must be greater than ${ minMemoryPerCore / (1024 * 1024) }MB.")

    val fullRowType = vsm.rvRowType

    val bpvType = BitPackedVectorView.rvRowType(vsm.rowKeyTypes(0), vsm.rowKeyTypes(1))

    val locusIndex = vsm.rowType.fieldIdx("locus")
    val allelesIndex = vsm.rowType.fieldIdx("alleles")

    val typ = vsm.rvd.typ

    val standardizedRDD = vsm.rvd
      .mapPartitionsPreservesPartitioning(new OrderedRVDType(typ.partitionKey, typ.key, bpvType))({ it =>
        val hcView = HardCallView(fullRowType)
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv2 = RegionValue(region)

        it.flatMap { rv =>
          region.clear()
          hcView.setRegion(rv)
          rvb.set(region)
          rvb.start(bpvType)
          rvb.startStruct()
          rvb.addField(fullRowType, rv, locusIndex)
          rvb.addField(fullRowType, rv, allelesIndex)

          val keep = addBitPackedVector(rvb, hcView, nSamples) // add bit packed genotype vector with metadata

          if (keep) {
            rvb.endStruct()
            rv2.setOffset(rvb.end())
            Some(rv2)
          }
          else
            None
        }
      })

    val ((rddLP1, nVariantsLP1, nPartitionsLP1), durationLP1) = time({
      val prunedRDD = pruneLocal(standardizedRDD, r2Threshold, windowSize, Option(maxQueueSize)).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      (prunedRDD, nVariantsKept, nPartitions)
    })
    info(s"LD prune step 1 of 3: nVariantsKept=$nVariantsLP1, nPartitions=$nPartitionsLP1, time=${ formatTime(durationLP1) }")

    val ((rddLP2, nVariantsLP2, nPartitionsLP2), durationLP2) = time({
      val (_, nPartitionsRequired) = estimateMemoryRequirements(nVariantsLP1, nSamples, nCores, memoryPerCore)
      val repartRDD = rddLP1.coalesce(nPartitionsRequired, shuffle = true).persist(StorageLevel.MEMORY_AND_DISK)
      repartRDD.count()
      rddLP1.unpersist()
      val prunedRDD = pruneLocal(repartRDD, r2Threshold, windowSize, None).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      repartRDD.unpersist()
      (prunedRDD, nVariantsKept, nPartitions)
    })
    info(s"LD prune step 2 of 3: nVariantsKept=$nVariantsLP2, nPartitions=$nPartitionsLP2, time=${ formatTime(durationLP2) }")

    val ((globalPrunedRDD, nVariantsFinal), globalDuration) = time(pruneGlobal(rddLP2, r2Threshold, windowSize))
    info(s"LD prune step 3 of 3: nVariantsKept=$nVariantsFinal, time=${ formatTime(globalDuration) }")

    vsm.copy2(rvd = vsm.rvd.copy(rdd = vsm.rvd.orderedJoinDistinct(globalPrunedRDD, "inner").map(_.rvLeft)))
  }
}
