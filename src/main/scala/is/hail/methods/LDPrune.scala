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

  case class GlobalPruneIntermediate(rdd: GeneralRDD[(Variant, BitPackedVector)], index: Int, persist: Boolean)

  case class BitPackedVector(gs: Array[Long], nSamples: Int, mean: Double, sdRecip: Double) {
    def unpack(): Array[Int] = {

      val shifts = (0 until genotypesPerPack * 2 by 2).reverse

      gs.flatMap { l =>
        shifts.map { s => (l >> s) & 3 match {
          case 0 => 0
          case 1 => 1
          case 2 => 2
          case 3 => -1
        }
        }
      }.take(nSamples)
    }
  }

  val table = {
    val t = Array.fill[Byte](256 * 4)(0)

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

  def findTableValue(a: Int, b: Int, c: Int, d: Int): (Int, Int, Int, Int) = {
    val r1 = findTableValue(a, b)
    val r2 = findTableValue(c, d)
    (r1._1 + r2._1, r1._2 + r2._2, r1._3 + r2._3, r1._4 + r2._4)
  }

  def findTableValue(xi: Int, yi: Int): (Int, Int, Int, Int) = {
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

  def pack(nGenotypes: Int, gs: Array[Byte]): Array[Long] = {
    require(nGenotypes % genotypesPerPack == 0)

    val nPacks = nGenotypes / genotypesPerPack
    val packedGenotypes = new Array[Long](nPacks)
    var pack = 0
    while (pack != nPacks) {
      val k = pack * genotypesPerPack

      packedGenotypes(pack) =
        gs(k).toLong << 62 | gs(k + 1).toLong << 60 | gs(k + 2).toLong << 58 | gs(k + 3).toLong << 56 |
          gs(k + 4).toLong << 54 | gs(k + 5).toLong << 52 | gs(k + 6).toLong << 50 | gs(k + 7).toLong << 48 |
          gs(k + 8).toLong << 46 | gs(k + 9).toLong << 44 | gs(k + 10).toLong << 42 | gs(k + 11).toLong << 40 |
          gs(k + 12).toLong << 38 | gs(k + 13).toLong << 36 | gs(k + 14).toLong << 34 | gs(k + 15).toLong << 32 |
          gs(k + 16).toLong << 30 | gs(k + 17).toLong << 28 | gs(k + 18).toLong << 26 | gs(k + 19).toLong << 24 |
          gs(k + 20).toLong << 22 | gs(k + 21).toLong << 20 | gs(k + 22).toLong << 18 | gs(k + 23).toLong << 16 |
          gs(k + 24).toLong << 14 | gs(k + 25).toLong << 12 | gs(k + 26).toLong << 10 | gs(k + 27).toLong << 8 |
          gs(k + 28).toLong << 6 | gs(k + 29).toLong << 4 | gs(k + 30).toLong << 2 | gs(k + 31).toLong

      pack += 1
    }
    packedGenotypes
  }

  def toBitPackedVector(gs: Iterable[Genotype], nSamples: Int): Option[BitPackedVector] = {
    val padding = genotypesPerPack - nSamples % genotypesPerPack
    val nGenotypes = nSamples + padding
    val nPacks = nGenotypes / genotypesPerPack
    require(nGenotypes % genotypesPerPack == 0)
    val a = new Array[Long](nPacks) // padded values are 0 by default which do not affect the result when computing r2
    toBitPackedVector(gs, nSamples, a)
  }

  def toBitPackedVector(gs: Iterable[Genotype], nSamples: Int, a: Array[Long]): Option[BitPackedVector] = {
    val it = gs.iterator
    val nBitsPerPack = 2 * genotypesPerPack

    var nPresent = 0
    var gtSum = 0
    var gtSumSq = 0

    var pack = 0L
    var packOffset = nBitsPerPack - 2
    var packIndex = 0
    var i = 0
    while (i < nSamples) {
      val gt = it.next().unboxedGT

      pack = pack | ((gt & 3).toLong << packOffset)

      if (packOffset == 0) {
        a(packIndex) = pack
        packIndex += 1
        pack = 0L
        packOffset = nBitsPerPack
      }
      packOffset -= 2

      if (gt >= 0) {
        nPresent += 1
        gtSum += gt
        gtSumSq += gt * gt
      }
      i += 1
    }

    a(packIndex) = pack

    val nMissing = nSamples - nPresent
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


  def r2(x: BitPackedVector, y: BitPackedVector): Double = {
    require(x.nSamples == y.nSamples)

    val gsX = x.gs
    val gsY = y.gs

    val N = x.nSamples
    val meanX = x.mean
    val meanY = y.mean
    val sdrecipX = x.sdRecip
    val sdrecipY = y.sdRecip

    var XbarYbarCount = 0
    var XbarCount = 0
    var YbarCount = 0
    var xySum = 0

    val nPacks = gsX.length
    var pack = 0
    while (pack < nPacks) {
      val lX = gsX(pack)
      val lY = gsY(pack)
      var shift = 2 * (genotypesPerPack - 2)

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

    val r = sdrecipX * sdrecipY * ((xySum + XbarCount * meanX + YbarCount * meanY + XbarYbarCount * meanX * meanY) - N * meanX * meanY)
    r * r
  }

  private def pruneLocal(inputRDD: OrderedRDD[Locus, Variant, BitPackedVector], r2Threshold: Double, window: Int, queueSize: Option[Int]) = {
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
          val (v2, bpv2) = qit.next()
          if (v.contig != v2.contig || v.start - v2.start > window)
            done = true
          else {
            if (r2(bpv, bpv2) >= r2Threshold) {
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

  private def pruneGlobal(inputRDD: OrderedRDD[Locus, Variant, BitPackedVector], r2Threshold: Double, window: Int) = {
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
        val minimumLocus = Locus(startLocus.contig, math.max(startLocus.position - window, 0))
        val partMinimum = rangePartitioner.getPartitionT(minimumLocus)
        partitionIndices.filter(idx => idx >= partMinimum && idx <= partitionId).reverse
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
          it.foreach { case (v2, bpv2) =>
            targetData = targetData.filter { case (v, bpv) =>
              if (v.contig != v2.contig || v.start - v2.start > window)
                true
              else {
                r2(bpv, bpv2) < r2Threshold
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

  def estimateMemoryRequirements(nVariants: Long, nSamples: Int, memoryPerCore: Long) = {
    val numBytesPerVariant = math.ceil(nSamples.toDouble / genotypesPerPack).toLong + variantByteOverhead
    val memoryAvailPerCore = memoryPerCore * fractionMemoryToUse

    val maxQueueSize = math.max(1, math.ceil(memoryAvailPerCore / numBytesPerVariant).toInt)
    val numPartitionsRequired = math.max(1, math.ceil(nVariants.toDouble / maxQueueSize).toInt)

    assert(maxQueueSize > 0 && numPartitionsRequired > 0)

    (maxQueueSize, numPartitionsRequired)
  }

  def ldPrune(vds: VariantDataset, r2Threshold: Double, window: Int, memoryPerCore: Long = 1073741824) = {
    if (r2Threshold < 0 || r2Threshold > 1)
      fatal(s"R^2 threshold must be in the range [0,1]. Found `$r2Threshold'.")

    if (window < 0)
      fatal(s"Window must be greater than or equal to 0. Found `$window'.")

    val sc = vds.sparkContext
    val partitionSizesInitial = sc.runJob(vds.rdd, getIteratorSize _)
    val nVariantsInitial = partitionSizesInitial.sum

    val nSamples = vds.nSamples
    val padding = genotypesPerPack - nSamples % genotypesPerPack
    val nGenotypes = nSamples + padding
    val nPacks = nGenotypes / genotypesPerPack
    require(nGenotypes % genotypesPerPack == 0)

    val minMemoryPerCore = math.ceil((1 / fractionMemoryToUse) * 8 * nSamples + variantByteOverhead)
    val (maxQueueSize, _) = estimateMemoryRequirements(nVariantsInitial, nSamples, memoryPerCore)

    info(s"InputData: nSamples=$nSamples nVariants=$nVariantsInitial nPartitions=${ partitionSizesInitial.length } maxQueueSize=$maxQueueSize")

    if (memoryPerCore < minMemoryPerCore)
      fatal(s"Memory per core must be greater than ${ minMemoryPerCore / (1024 * 1024) }MB")

    val standardizedRDD = vds.filterVariants { case (v, va, gs) => v.isBiallelic }.rdd
      .mapPartitions { it =>
        val a = new Array[Long](nGenotypes) // padded values are 0 by default which do not affect the result when computing r2
        it.flatMap { case (v, (va, gs)) =>
          toBitPackedVector(gs, nSamples, a) match {
            case Some(bpv) => Some((v, bpv))
            case None => None
          }
        }
      }.toOrderedRDD

//
//        val standardizedRDD2 = vds.filterVariants { case (v, va, gs) => v.isBiallelic }.rdd
//          .flatMapValues { case (va, gs) => toBitPackedVector(gs, nSamples) }.asOrderedRDD

    val ((rddLP1, nVariantsLP1, nPartitionsLP1), durationLP1) = time({
      val prunedRDD = pruneLocal(standardizedRDD, r2Threshold, window, Option(maxQueueSize)).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      standardizedRDD.unpersist()
      (prunedRDD, nVariantsKept, nPartitions)
    })
    info(s"Local Prune 1: nVariantsKept=$nVariantsLP1 nPartitions=$nPartitionsLP1 time=${ formatTime(durationLP1) }")


    val ((rddLP2, nVariantsLP2, nPartitionsLP2), durationLP2) = time({
      val nPartitionsRequired = estimateMemoryRequirements(nVariantsLP1, nSamples, memoryPerCore)._2
      val repartRDD = rddLP1.coalesce(nPartitionsRequired, shuffle = true)(null).asOrderedRDD
      repartRDD.persist(StorageLevel.MEMORY_AND_DISK)
      repartRDD.count()
      rddLP1.unpersist()
      val prunedRDD = pruneLocal(repartRDD, r2Threshold, window, None).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      repartRDD.unpersist()
      (prunedRDD, nVariantsKept, nPartitions)
    })
    info(s"Local Prune 2: nVariantsKept=$nVariantsLP2 nPartitions=$nPartitionsLP2 time=${ formatTime(durationLP2) }")


    val ((globalPrunedRDD, nVariantsFinal), globalDuration) = time(pruneGlobal(rddLP2, r2Threshold, window))
    info(s"Global Prune: nVariantsRemaining=$nVariantsFinal time=${ formatTime(globalDuration) }")

    vds.copy(rdd = vds.rdd.orderedInnerJoinDistinct(globalPrunedRDD)
      .mapValues { case ((va, gs), _) => (va, gs) }
      .asOrderedRDD)
  }
}
