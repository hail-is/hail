package is.hail.methods

import java.util

import breeze.linalg.{Vector => BVector}
import is.hail.annotations.Annotation
import is.hail.sparkextras.GeneralRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import is.hail.sparkextras._
import is.hail.variant._
import is.hail.utils._

case class GlobalPruneIntermediate(rdd: GeneralRDD[(Variant, BVector[Double])], index: Int, persist: Boolean)

case class BitVectorInput(gs: Array[Long], nSamples: Int, mean: Double, sdRecip: Double)

object LDPrune {
  val variantByteOverhead = 50
  val fractionMemoryToUse = 0.25
  val genotypesPerPack = 32

  def pack(nGenotypes: Int, gs: Array[Byte]): Array[Long] = { // Each byte is one person's genotype (0, 1, 2, 3); made sure nGenotypes is mult 32 earlier
    require(nGenotypes % genotypesPerPack == 0)

    val nPacks = nGenotypes / genotypesPerPack // number of longs
    val packedGenotypes = new Array[Long](nPacks)
    var pack = 0
    while (pack != nPacks) {
      val k = pack * genotypesPerPack

      packedGenotypes(pack) =
          gs(k).toLong       << 62 | gs(k + 1 ).toLong  << 60 | gs(k + 2 ).toLong  << 58 | gs(k + 3 ).toLong  << 56 |
          gs(k + 4 ).toLong  << 54 | gs(k + 5 ).toLong  << 52 | gs(k + 6 ).toLong  << 50 | gs(k + 7 ).toLong  << 48 |
          gs(k + 8 ).toLong  << 46 | gs(k + 9 ).toLong  << 44 | gs(k + 10 ).toLong << 42 | gs(k + 11 ).toLong << 40 |
          gs(k + 12 ).toLong << 38 | gs(k + 13 ).toLong << 36 | gs(k + 14 ).toLong << 34 | gs(k + 15 ).toLong << 32 |
          gs(k + 16 ).toLong << 30 | gs(k + 17 ).toLong << 28 | gs(k + 18 ).toLong << 26 | gs(k + 19 ).toLong << 24 |
          gs(k + 20 ).toLong << 22 | gs(k + 21 ).toLong << 20 | gs(k + 22 ).toLong << 18 | gs(k + 23 ).toLong << 16 |
          gs(k + 24 ).toLong << 14 | gs(k + 25 ).toLong << 12 | gs(k + 26 ).toLong << 10 | gs(k + 27 ).toLong << 8  |
          gs(k + 28 ).toLong << 6  | gs(k + 29 ).toLong << 4  | gs(k + 30 ).toLong << 2  | gs(k + 31 ).toLong

      pack += 1
    }
    packedGenotypes
  }

  def unpack(gs: Array[Long]): Array[Int] = {
    val shifts = (0 until genotypesPerPack * 2 by 2).reverse
    gs.flatMap{ l => shifts.map(s => l >> s & 3 match {
      case 0 => 0
      case 1 => 1
      case 2 => -1
      case 3 => 2
      case _ => fatal("genotype unpacking not correct")
    })}
  }

  def r2(x: BitVectorInput, y: BitVectorInput): Double = {
    require(x.nSamples == y.nSamples)

    val gtsX = unpack(x.gs)
    val gtsY = unpack(y.gs)

    val N = x.nSamples
    val meanX = x.mean
    val meanY = y.mean
    val sdrecipX = x.sdRecip
    val sdrecipY = y.sdRecip

    var XbarYbarCount = 0
    var XbarCount = 0
    var YbarCount = 0
    var xySum = 0

    gtsX.zip(gtsY).foreach { case (gtX, gtY) =>
      (gtX, gtY) match {
        case (-1, -1) => XbarYbarCount += 1
        case (-1, _) => XbarCount += gtY
        case (_, -1) => YbarCount += gtX
        case (_, _) => xySum += gtX * gtY
      }
     }

    val r = sdrecipX * sdrecipY * ((xySum + XbarCount * meanX + YbarCount * meanY + XbarYbarCount * meanX * meanY) - N * meanX * meanY)
    r * r
  }

  def toByteGtArray(gs: Iterable[Genotype], nSamples: Int): Option[BitVectorInput] = {
    val padding = genotypesPerPack - nSamples % genotypesPerPack
    val nGenotypes = nSamples + padding

    val gts = new Array[Byte](nGenotypes)
    val it = gs.iterator

    var nPresent = 0
    var gtSum = 0
    var gtSumSq = 0

    var i = 0
    while (i < nSamples) {
      val gt = it.next().unboxedGT
      val gtInt = gt match {
        case -1 => 2
        case 0 => 0
        case 1 => 1
        case 2 => 3
      }

      gts.update(i, gtInt.toByte)
      if (gt >= 0) {
        nPresent += 1
        gtSum += gt
        gtSumSq += gt * gt
      }
      i += 1
    }

    val nMissing = nSamples - nPresent
    val allHomRef = gtSum == 0
    val allHet = gtSum == nPresent && gtSumSq == nPresent
    val allHomVar = gtSum == 2 * nPresent

    if (allHomRef || allHet || allHomVar || nMissing == nSamples)
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      val gtMeanSq = (gtSumSq + nMissing * gtMean * gtMean) / nSamples
      val gtStdDevRec = 1d / math.sqrt((gtMeanSq - gtMean * gtMean) * nSamples)

      Some(BitVectorInput(pack(nGenotypes, gts), nSamples, gtMean, gtStdDevRec))
    }
  }

  def toNormalizedGtArray(gs: Iterable[Genotype], nSamples: Int): Option[Array[Double]] = {
    val a = new Array[Double](nSamples)
    val gts = new Array[Int](nSamples)
    val it = gs.iterator

    var nPresent = 0
    var gtSum = 0
    var gtSumSq = 0

    var i = 0
    while (i < nSamples) {
      val gt = it.next().unboxedGT
      gts.update(i, gt)
      if (gt >= 0) {
        nPresent += 1
        gtSum += gt
        gtSumSq += gt * gt
      }
      i += 1
    }

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

      var i = 0
      while (i < nSamples) {
        val gt = gts(i)
        if (gt >= 0)
          a.update(i, (gt - gtMean) * gtStdDevRec)
        i += 1
      }

      println(s"gtSum = $gtSum")
      println(s"gtSumSq = $gtSumSq")
      println(s"nPresent = $nPresent")
      println(s"nSamples = $nSamples")
      println(s"nMissing = $nMissing")
      println(s"gtMean = $gtMean")
      println(s"gtMeanAll = $gtMeanAll")
      println(s"gtMeanAllSq = $gtMeanSqAll")
      println(s"gtStdDevRec = $gtStdDevRec")
      println(s"a = ${a.mkString(",")}")

      Some(a)
    }
  }

  private def pruneLocal(inputRDD: OrderedRDD[Locus, Variant, BVector[Double]], r2Threshold: Double, window: Int, queueSize: Option[Int]) = {
    inputRDD.rdd.mapPartitions({ it =>
      val queue = queueSize match {
        case Some(qs) => new util.ArrayDeque[(Variant, BVector[Double])](qs)
        case None => new util.ArrayDeque[(Variant, BVector[Double])]
      }

      it.filter { case (v, sgs) =>
        var keepVariant = true
        var done = false
        val qit = queue.descendingIterator()

        while (!done && qit.hasNext) {
          val (v2, sgs2) = qit.next()
          if (v.contig != v2.contig || v.start - v2.start > window)
            done = true
          else {
            val r = sgs.dot(sgs2)
            if ((r * r: Double) >= r2Threshold) {
              keepVariant = false
              done = true
            }
          }
        }

        if (keepVariant) {
          queue.addLast((v, sgs))
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

  private def pruneGlobal(inputRDD: OrderedRDD[Locus, Variant, BVector[Double]], r2Threshold: Double, window: Int) = {
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

    def pruneF = (x: Array[Iterator[(Variant, BVector[Double])]]) => {
      val nPartitions = x.length
      val targetIterator = x(0)
      val prevPartitions = x.drop(1).reverse

      if (nPartitions == 1)
        targetIterator
      else {
        var targetData = targetIterator.toArray

        prevPartitions.foreach { it =>
          it.foreach { case (v2, sgs2) =>
            targetData = targetData.filter { case (v, sgs) =>
              if (v.contig != v2.contig || v.start - v2.start > window)
                true
              else {
                val r = sgs.dot(sgs2)
                (r * r: Double) < r2Threshold
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

    def generalRDDInputs(partitionIndex: Int): (Array[RDD[(Variant, BVector[Double])]], Array[(Int, Int)]) = {
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

    val prunedRDD = new GeneralRDD[(Variant, BVector[Double])](sc, pruneIntermediates.map(_.rdd),
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
    val numBytesPerVariant = 8 * nSamples + variantByteOverhead
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
    val nSamples = vds.nSamples
    val partitionSizesInitial = sc.runJob(vds.rdd, getIteratorSize _)
    val nVariantsInitial = partitionSizesInitial.sum

    val minMemoryPerCore = math.ceil((1 / fractionMemoryToUse) * 8 * nSamples + variantByteOverhead)
    val (maxQueueSize, _) = estimateMemoryRequirements(nVariantsInitial, nSamples, memoryPerCore)

    info(s"InputData: nSamples=$nSamples nVariants=$nVariantsInitial nPartitions=${ partitionSizesInitial.length } maxQueueSize=$maxQueueSize")

    if (memoryPerCore < minMemoryPerCore)
      fatal(s"Memory per core must be greater than ${ minMemoryPerCore / (1024 * 1024) }MB")

    val repartitionRequired = partitionSizesInitial.exists(_ > maxQueueSize)

    val standardizedRDD = vds.rdd.flatMapValues { case (va, gs) =>
      toNormalizedGtArray(gs, nSamples).map(BVector(_))
    }.asOrderedRDD

    val ((rddLP1, nVariantsLP1, nPartitionsLP1), durationLP1) = time({
      val prunedRDD = pruneLocal(standardizedRDD, r2Threshold, window, Option(maxQueueSize)).persist(StorageLevel.MEMORY_AND_DISK)
      val nVariantsKept = prunedRDD.count()
      val nPartitions = prunedRDD.partitions.length
      assert(nVariantsKept >= 1)
      standardizedRDD.unpersist()
      (prunedRDD, nVariantsKept, nPartitions)
    })
    info(s"Local Prune 1: nVariantsKept=$nVariantsLP1 nPartitions=$nPartitionsLP1 time=${ formatTime(durationLP1) }")

    val localPrunedRDD =
      if (!repartitionRequired) {
        rddLP1
      } else {
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
        rddLP2
      }

    val ((globalPrunedRDD, nVariantsFinal), globalDuration) = time(pruneGlobal(localPrunedRDD, r2Threshold, window))
    info(s"Global Prune: nVariantsRemaining=$nVariantsFinal time=${ formatTime(globalDuration) }")

    vds.copy(rdd = vds.rdd.orderedInnerJoinDistinct(globalPrunedRDD)
      .mapValues { case ((va, gs), _) => (va, gs) }
      .asOrderedRDD)
  }
}
