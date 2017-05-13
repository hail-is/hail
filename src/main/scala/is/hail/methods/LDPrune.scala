package is.hail.methods

import java.util

import is.hail.annotations.Annotation
import is.hail.sparkextras.GeneralRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import is.hail.sparkextras._
import is.hail.variant._
import is.hail.utils._

object LDPrune {
  val variantByteOverhead = 50
  val fractionMemoryToUse = 0.25
  val genotypesPerPack = 32
  val nPartitionsPerCore = 3

  case class GlobalPruneIntermediate(rdd: GeneralRDD[(Variant, BitPackedVector)], index: Int, persist: Boolean)

  case class BitPackedVector(gs: Array[Long], nSamples: Int, mean: Double, stdDevRec: Double) {
    def unpack(): Array[Int] = {
      val gts = Array.ofDim[Int](nSamples)
      val nPacks = gs.length

      var packIndex = 0
      var i = 0
      val shiftInit = genotypesPerPack * 2 - 2
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

  def toBitPackedVector(it: HailIterator[Int], nSamples: Int): Option[BitPackedVector] = {
    require(nSamples >= 0)
    val nBitsPerPack = 2 * genotypesPerPack
    val nPacks = (nSamples - 1) / genotypesPerPack + 1
    val a = Array.ofDim[Long](nPacks)

    var nMissing = 0
    var gtSum = 0
    var gtSumSq = 0

    var pack = 0L
    var packOffset = nBitsPerPack - 2
    var packIndex = 0
    var i = 0
    while (i < nSamples) {
      val gt = it.next()

      pack = pack | ((gt & 3).toLong << packOffset)

      if (packOffset == 0) {
        a(packIndex) = pack
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
      a(packIndex) = pack

    val nPresent = nSamples - nMissing
    val allHomRef = gtSum == 0
    val allHet = gtSum == nPresent && gtSumSq == nPresent
    val allHomVar = gtSum == 2 * nPresent

    if (allHomRef || allHet || allHomVar || nMissing == nSamples)
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      val gtMeanAll = (gtSum + nMissing * gtMean) / nSamples
      val gtMeanSqAll = (gtSumSq + nMissing * gtMean * gtMean) / nSamples
      val gtStdDevRec = 1d / math.sqrt((gtMeanSqAll - gtMeanAll * gtMeanAll) * nSamples)

      Some(BitPackedVector(a, nSamples, gtMean, gtStdDevRec))
    }
  }

  def computeR2(x: BitPackedVector, y: BitPackedVector): Double = {
    require(x.nSamples == y.nSamples)

    val gsX = x.gs
    val gsY = y.gs

    val N = x.nSamples
    val meanX = x.mean
    val meanY = y.mean
    val stdDevRecX = x.stdDevRec
    val stdDevRecY = y.stdDevRec

    var XbarYbarCount = 0
    var XbarCount = 0
    var YbarCount = 0
    var xySum = 0

    val nPacks = gsX.length
    val shiftInit = 2 * (genotypesPerPack - 2)
    var pack = 0
    while (pack < nPacks) {
      val lX = gsX(pack)
      val lY = gsY(pack)
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
    val r2 = r * r
    assert(D_>=(r2, 0d) && D_<=(r2, 1d), s"R2 must lie in [0,1]. Found $r2.")
    r2
  }

  private def pruneLocal(inputRDD: OrderedRDD[Locus, Variant, BitPackedVector],
    r2Threshold: Double, windowSize: Int, queueSize: Option[Int]): OrderedRDD[Locus, Variant, BitPackedVector] = {
    inputRDD.rdd.mapPartitions({ it =>
      val queue = queueSize match {
        case Some(qs) => new util.ArrayDeque[(Variant, BitPackedVector)](qs)
        case None => new util.ArrayDeque[(Variant, BitPackedVector)]
      }

      it.filter { case (v, bpv) =>
        var keepVariant = true
        var done = false
        val qit = queue.descendingIterator()

        while (!done && qit.hasNext) {
          val (vPrev, bpvPrev) = qit.next()
          if (v.contig != vPrev.contig || v.start - vPrev.start > windowSize)
            done = true
          else {
            val r2 = computeR2(bpv, bpvPrev)
            if (r2 >= r2Threshold) {
              keepVariant = false
              done = true
            }
          }
        }

        if (keepVariant) {
          queue.addLast((v, bpv))
          queueSize.foreach { qs =>
            if (queue.size() > qs) {
              queue.pop()
            }
          }
        }

        keepVariant
      }
    }, preservesPartitioning = true).asOrderedRDD
  }

  private def pruneGlobal(inputRDD: OrderedRDD[Locus, Variant, BitPackedVector],
    r2Threshold: Double, windowSize: Int): (OrderedRDD[Locus, Variant, Any], Long) = {
    val sc = inputRDD.sparkContext

    require(sc.getPersistentRDDs.get(inputRDD.id).isDefined)

    val rangePartitioner = inputRDD.orderedPartitioner
    val rangeBounds = rangePartitioner.rangeBounds
    val partitionIndices = inputRDD.getPartitions.map(_.index)
    val nPartitions = inputRDD.partitions.length

    def computeDependencies(partitionId: Int): Array[Int] = {
      if (partitionId == partitionIndices(0))
        Array(partitionId)
      else {
        val startLocus = rangeBounds(partitionId - 1) // this is the best estimate of the first locus of the partition
        val minLocus = Locus(startLocus.contig, math.max(startLocus.position - windowSize, 0))
        val minPart = rangePartitioner.getPartitionT(minLocus)
        partitionIndices.filter(idx => idx >= minPart && idx <= partitionId).reverse
      }
    }

    def pruneF = (x: Array[Iterator[(Variant, BitPackedVector)]]) => {
      val nPartitions = x.length
      val targetIterator = x(0)
      val prevPartitions = x.drop(1).reverse

      if (nPartitions == 1)
        targetIterator
      else {
        var targetData = targetIterator.toArray

        prevPartitions.foreach { it =>
          it.foreach { case (vPrev, bpvPrev) =>
            targetData = targetData.filter { case (v, bpv) =>
              if (v.contig != vPrev.contig || math.abs(v.start - vPrev.start) > windowSize)
                true
              else {
                computeR2(bpv, bpvPrev) < r2Threshold
              }
            }
          }
        }
        targetData.iterator
      }
    }

    val contigStartPartitions =
      partitionIndices.filter { i =>
        i == partitionIndices(0) || i == partitionIndices.last || rangeBounds(i).contig != rangeBounds(i - 1).contig
      }

    val pruneIntermediates = Array.fill[GlobalPruneIntermediate](nPartitions)(null)

    def generalRDDInputs(partitionIndex: Int): (Array[RDD[(Variant, BitPackedVector)]], Array[(Int, Int)]) = {
      val (rdds, inputs) = computeDependencies(partitionIndex).zipWithIndex.map { case (depIndex, i) =>
        if (depIndex == partitionIndex || contigStartPartitions.contains(depIndex))
          (inputRDD, (i, depIndex))
        else {
          val gpi = pruneIntermediates(depIndex)
          pruneIntermediates(depIndex) = gpi.copy(persist = true)
          (gpi.rdd, (i, gpi.index))
        }
      }.unzip
      (rdds.toArray, inputs.toArray)
    }

    for (i <- partitionIndices) {
      val (rdds, inputs) = generalRDDInputs(i)
      pruneIntermediates(i) = GlobalPruneIntermediate(rdd = new GeneralRDD(sc, rdds, Array((inputs, pruneF))), index = 0, persist = false) // creating single partition RDDs with partition index = 0
    }

    pruneIntermediates.foreach { gpi =>
      if (gpi.persist) gpi.rdd.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val prunedRDD = new GeneralRDD[(Variant, BitPackedVector)](sc, pruneIntermediates.map(_.rdd),
      pruneIntermediates.zipWithIndex.map { case (gpi, i) =>
        (Array((i, gpi.index)), pruneF)
      })

    val annotRDD = prunedRDD.mapValues(_ => Annotation(true)).toOrderedRDD.persist(StorageLevel.MEMORY_AND_DISK)
    val nVariantsKept = annotRDD.count()

    pruneIntermediates.foreach { gpi => gpi.rdd.unpersist() }
    inputRDD.unpersist()

    (annotRDD, nVariantsKept)
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

  def apply(vds: VariantDataset, r2Threshold: Double, windowSize: Int, nCores: Int, memoryPerCore: Long = 268435456): VariantDataset = {
    if (nCores <= 0)
      fatal(s"Number of cores must be positive.")

    if (r2Threshold < 0 || r2Threshold > 1)
      fatal(s"R^2 threshold must be in the range [0,1]. Found `$r2Threshold'.")

    if (windowSize < 0)
      fatal(s"Window size must be greater than or equal to 0. Found `$windowSize'.")

    val sc = vds.sparkContext

    val nVariantsInitial = vds.countVariants()
    val nPartitionsInitial = vds.nPartitions
    val nSamples = vds.nSamples

    val minMemoryPerCore = math.ceil((1d / fractionMemoryToUse) * 8 * nSamples + variantByteOverhead)
    val (maxQueueSize, _) = estimateMemoryRequirements(nVariantsInitial, nSamples, nCores, memoryPerCore)

    info(s"Running LD prune with nSamples=$nSamples, nVariants=$nVariantsInitial, nPartitions=$nPartitionsInitial, and maxQueueSize=$maxQueueSize.")

    if (memoryPerCore < minMemoryPerCore)
      fatal(s"`memory_per_core' must be greater than ${ minMemoryPerCore / (1024 * 1024) }MB.")

    val standardizedRDD: OrderedRDD[Locus, Variant, BitPackedVector] = vds.filterVariants { case (v, va, gs) => v.isBiallelic }.rdd
      .flatMapValues { case (va, gs) => toBitPackedVector(gs.hardCallIterator, nSamples) }.asOrderedRDD

    val ((rddLP1, nVariantsLP1, nPartitionsLP1), durationLP1) = time({
      val prunedRDD = pruneLocal(standardizedRDD, r2Threshold, windowSize, Option(maxQueueSize)).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      standardizedRDD.unpersist()
      (prunedRDD, nVariantsKept, nPartitions)
    })
    info(s"LD prune step 1 of 3: nVariantsKept=$nVariantsLP1, nPartitions=$nPartitionsLP1, time=${ formatTime(durationLP1) }")

    val ((rddLP2, nVariantsLP2, nPartitionsLP2), durationLP2) = time({
      val (_, nPartitionsRequired) = estimateMemoryRequirements(nVariantsLP1, nSamples, nCores, memoryPerCore)
      val repartRDD = rddLP1.coalesce(nPartitionsRequired, shuffle = true)(null).asOrderedRDD
      repartRDD.persist(StorageLevel.MEMORY_AND_DISK)
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

    vds.copy(rdd = vds.rdd.orderedInnerJoinDistinct(globalPrunedRDD)
      .mapValues { case ((va, gs), _) => (va, gs) }
      .asOrderedRDD)
  }
}
