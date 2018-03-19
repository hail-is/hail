package is.hail.rvd

import java.util

import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.reflect.ClassTag

class OrderedRVD(
  val typ: OrderedRVDType,
  val partitioner: OrderedRVDPartitioner,
  val crdd: ContextRDD[RVDContext, RegionValue]) extends RVD {
  self =>

  val rdd = crdd.run

  def rowType: TStruct = typ.rowType

  def updateType(newTyp: OrderedRVDType): OrderedRVD =
    OrderedRVD(newTyp, partitioner, rdd)

  def mapPreservesPartitioning(newTyp: OrderedRVDType)(f: (RegionValue) => RegionValue): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      rdd.map(f))

  def mapPartitionsWithIndexPreservesPartitioning(newTyp: OrderedRVDType)(f: (Int, Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      rdd.mapPartitionsWithIndex(f))

  def mapPartitionsPreservesPartitioning(newTyp: OrderedRVDType)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      rdd.mapPartitions(f))

  override def filter(p: (RegionValue) => Boolean): OrderedRVD =
    OrderedRVD(typ,
      partitioner,
      rdd.filter(p))

  def sample(withReplacement: Boolean, p: Double, seed: Long): OrderedRVD =
    OrderedRVD(typ,
      partitioner,
      rdd.sample(withReplacement, p, seed))

  def persist(level: StorageLevel): OrderedRVD = {
    val PersistedRVRDD(persistedRDD, iterationRDD) = persistRVRDD(level)
    new OrderedRVD(typ, partitioner, iterationRDD) {
      override def storageLevel: StorageLevel = persistedRDD.getStorageLevel

      override def persist(newLevel: StorageLevel): OrderedRVD = {
        if (newLevel == StorageLevel.NONE)
          unpersist()
        else {
          persistedRDD.persist(newLevel)
          this
        }
      }

      override def unpersist(): OrderedRVD = {
        persistedRDD.unpersist()
        self
      }
    }
  }

  override def cache(): OrderedRVD = persist(StorageLevel.MEMORY_ONLY)

  override def unpersist(): OrderedRVD = this

  def constrainToOrderedPartitioner(
    ordType: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner
  ): OrderedRVD = {

    require(ordType.rowType == typ.rowType)
    require(ordType.kType isPrefixOf typ.kType)
    require(newPartitioner.pkType isIsomorphicTo ordType.pkType)
    // Should remove this requirement in the future
    require(typ.pkType isPrefixOf ordType.pkType)

    new OrderedRVD(
      typ = ordType,
      partitioner = newPartitioner,
      rdd = RepartitionedOrderedRDD(this, newPartitioner))
  }

  def keyBy(key: Array[String] = typ.key): KeyedOrderedRVD =
    new KeyedOrderedRVD(this, key)

  def orderedJoin(
    right: OrderedRVD,
    joinType: String,
    joiner: Iterator[JoinedRegionValue] => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD =
    keyBy().orderedJoin(right.keyBy(), joinType, joiner, joinedType)

  def orderedJoinDistinct(
    right: OrderedRVD,
    joinType: String,
    joiner: Iterator[JoinedRegionValue] => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD =
    keyBy().orderedJoinDistinct(right.keyBy(), joinType, joiner, joinedType)

  def orderedZipJoin(right: OrderedRVD): RDD[JoinedRegionValue] =
    keyBy().orderedZipJoin(right.keyBy())

  def partitionSortedUnion(rdd2: OrderedRVD): OrderedRVD = {
    assert(typ == rdd2.typ)
    assert(partitioner == rdd2.partitioner)

    val localTyp = typ
    OrderedRVD(typ, partitioner,
      rdd.zipPartitions(rdd2.rdd) { case (it, it2) =>
        new Iterator[RegionValue] {
          private val bit = it.buffered
          private val bit2 = it2.buffered

          def hasNext: Boolean = bit.hasNext || bit2.hasNext

          def next(): RegionValue = {
            if (!bit.hasNext)
              bit2.next()
            else if (!bit2.hasNext)
              bit.next()
            else {
              val c = localTyp.kInRowOrd.compare(bit.head, bit2.head)
              if (c < 0)
                bit.next()
              else
                bit2.next()
            }
          }
        }
      })
  }

  def copy(typ: OrderedRVDType = typ,
    orderedPartitioner: OrderedRVDPartitioner = partitioner,
    rdd: RDD[RegionValue] = rdd): OrderedRVD = {
    OrderedRVD(typ, orderedPartitioner, rdd)
  }

  def blockCoalesce(partitionEnds: Array[Int]): OrderedRVD = {
    assert(partitionEnds.last == partitioner.numPartitions - 1 && partitionEnds(0) >= 0)
    assert(partitionEnds.zip(partitionEnds.tail).forall { case (i, inext) => i < inext })
    OrderedRVD(typ, partitioner.coalesceRangeBounds(partitionEnds), new BlockedRDD(rdd, partitionEnds))
  }

  def naiveCoalesce(maxPartitions: Int): OrderedRVD = {
    val n = partitioner.numPartitions
    if (maxPartitions >= n)
      return this

    val newN = maxPartitions
    val newNParts = Array.tabulate(newN)(i => (n - i + newN - 1) / newN)
    assert(newNParts.sum == n)
    assert(newNParts.forall(_ > 0))
    blockCoalesce(newNParts.scanLeft(-1)(_ + _).tail)
  }

  override def coalesce(maxPartitions: Int, shuffle: Boolean): OrderedRVD = {
    require(maxPartitions > 0, "cannot coalesce to nPartitions <= 0")
    val n = rdd.partitions.length
    if (!shuffle && maxPartitions >= n)
      return this
    if (shuffle) {
      val shuffled = rdd.coalesce(maxPartitions, shuffle = true)
      val ranges = OrderedRVD.calculateKeyRanges(typ, OrderedRVD.getPartitionKeyInfo(typ, OrderedRVD.getKeys(typ, shuffled)), shuffled.getNumPartitions)
      OrderedRVD.shuffle(typ, new OrderedRVDPartitioner(typ.partitionKey, typ.kType, ranges), shuffled)
    } else {

      val partSize = rdd.context.runJob(rdd, getIteratorSize _)
      log.info(s"partSize = ${ partSize.toSeq }")

      val partCumulativeSize = mapAccumulate[Array, Long](partSize, 0L)((s, acc) => (s + acc, s + acc))
      val totalSize = partCumulativeSize.last

      var newPartEnd = (0 until maxPartitions).map { i =>
        val t = totalSize * (i + 1) / maxPartitions

        /* j largest index not greater than t */
        var j = util.Arrays.binarySearch(partCumulativeSize, t)
        if (j < 0)
          j = -j - 1
        while (j < partCumulativeSize.length - 1
          && partCumulativeSize(j + 1) == t)
          j += 1
        assert(t <= partCumulativeSize(j) &&
          (j == partCumulativeSize.length - 1 ||
            t < partCumulativeSize(j + 1)))
        j
      }.toArray

      newPartEnd = newPartEnd.zipWithIndex.filter { case (end, i) => i == 0 || newPartEnd(i) != newPartEnd(i - 1) }
        .map(_._1)

      if (newPartEnd.length < maxPartitions)
        warn(s"coalesced to ${ newPartEnd.length } ${ plural(newPartEnd.length, "partition") }, less than requested $maxPartitions")

      blockCoalesce(newPartEnd)
    }
  }

  def filterIntervals(intervals: IntervalTree[_]): OrderedRVD = {
    val pkOrdering = typ.pkType.ordering
    val intervalsBc = rdd.sparkContext.broadcast(intervals)
    val rowType = typ.rowType
    val pkRowFieldIdx = typ.pkRowFieldIdx

    val pred: (RegionValue) => Boolean = (rv: RegionValue) => {
      val ur = new UnsafeRow(rowType, rv)
      val pk = Row.fromSeq(
        pkRowFieldIdx.map(i => ur.get(i)))
      intervalsBc.value.contains(pkOrdering, pk)
    }

    val nPartitions = partitions.length
    if (nPartitions <= 1)
      return filter(pred)

    val newPartitionIndices = intervals.toIterator.flatMap { case (i, _) =>
      if (!partitioner.rangeTree.probablyOverlaps(pkOrdering, i))
        IndexedSeq()
      else {
        val start = partitioner.getPartitionPK(i.start)
        val end = partitioner.getPartitionPK(i.end)
        start to end
      }
    }
      .toSet // distinct
      .toArray.sorted[Int]

    info(s"interval filter loaded ${ newPartitionIndices.length } of $nPartitions partitions")

    if (newPartitionIndices.isEmpty)
      OrderedRVD.empty(sparkContext, typ)
    else {
      val sub = subsetPartitions(newPartitionIndices)
      sub.copy(rdd = sub.rdd.filter(pred))
    }
  }

  def head(n: Long): OrderedRVD = {
    require(n >= 0)

    if (n == 0)
      return OrderedRVD.empty(sparkContext, typ)

    val newRDD = rdd.head(n)
    val newNParts = newRDD.getNumPartitions
    assert(newNParts > 0)

    val newRangeBounds = Array.range(0, newNParts).map(partitioner.rangeBounds)
    val newPartitioner = new OrderedRVDPartitioner(partitioner.partitionKey,
      partitioner.kType,
      newRangeBounds)

    OrderedRVD(typ, newPartitioner, newRDD)
  }

  def groupByKey(valuesField: String = "values"): OrderedRVD = {
    val newTyp = new OrderedRVDType(
      typ.partitionKey,
      typ.key,
      typ.kType ++ TStruct(valuesField -> TArray(typ.valueType)))
    val newRowType = newTyp.rowType

    val localType = typ

    val newRDD: RDD[RegionValue] = rdd.mapPartitions { it =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val outRV = RegionValue(region)
      val buffer = new RegionValueArrayBuffer(localType.valueType)
      val stepped: FlipbookIterator[FlipbookIterator[RegionValue]] =
        OrderedRVIterator(localType, it).staircase

      stepped.map { stepIt =>
        region.clear()
        buffer.clear()
        rvb.start(newRowType)
        rvb.startStruct()
        var i = 0
        while (i < localType.kType.size) {
          rvb.addField(localType.rowType, stepIt.value, localType.kRowFieldIdx(i))
          i += 1
        }
        for (rv <- stepIt)
          buffer.appendSelect(localType.rowType, localType.valueFieldIdx, rv)
        rvb.startArray(buffer.length)
        for (rv <- buffer)
          rvb.addRegionValue(localType.valueType, rv)
        rvb.endArray()
        rvb.endStruct()
        outRV.setOffset(rvb.end())
        outRV
      }
    }

    OrderedRVD(newTyp, partitioner, newRDD)
  }

  def distinctByKey(): OrderedRVD = {
    val localType = typ
    val newRVD = rdd.mapPartitions { it =>
      OrderedRVIterator(localType, it)
        .staircase
        .map(_.value)
    }
    OrderedRVD(typ, partitioner, newRVD)
  }

  def subsetPartitions(keep: Array[Int]): OrderedRVD = {
    require(keep.length <= rdd.partitions.length, "tried to subset to more partitions than exist")
    require(keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < rdd.partitions.length)),
      "values not sorted or not in range [0, number of partitions)")

    val newRangeBounds = Array.tabulate(keep.length) { i =>
      if (i == 0)
        partitioner.rangeBounds(keep(i))
      else {
        partitioner.rangeBounds(keep(i))
          .copy(start = partitioner.rangeBounds(keep(i - 1)).end)
      }
    }
    val newPartitioner = new OrderedRVDPartitioner(
      partitioner.partitionKey,
      partitioner.kType,
      newRangeBounds)

    OrderedRVD(typ, newPartitioner, rdd.subsetPartitions(keep))
  }

  def write(path: String, codecSpec: CodecSpec): Array[Long] = {
    val (partFiles, partitionCounts) = rdd.writeRows(path, rowType, codecSpec)
    val spec = OrderedRVDSpec(
      typ,
      codecSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(partitioner.rangeBounds, partitioner.rangeBoundsType))
    spec.write(sparkContext.hadoopConfiguration, path)
    partitionCounts
  }

  def zipPartitionsPreservesPartitioning[T: ClassTag](
    newTyp: OrderedRVDType,
    that: RDD[T]
  )(zipper: (Iterator[RegionValue], Iterator[T]) => Iterator[RegionValue]
  ): OrderedRVD =
    OrderedRVD(
      newTyp,
      partitioner,
      this.rdd.zipPartitions(that, preservesPartitioning = true)(zipper))

  def zipPartitionsPreservesPartitioning(
    newTyp: OrderedRVDType,
    that: RVD
  )(zipper: (Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD =
    OrderedRVD(
      newTyp,
      partitioner,
      this.rdd.zipPartitions(that.rdd, preservesPartitioning = true)(zipper))

  def writeRowsSplit(
    path: String,
    t: MatrixType,
    codecSpec: CodecSpec
  ): Array[Long] = rdd.writeRowsSplit(path, t, codecSpec, partitioner)
}

object OrderedRVD {
  def empty(sc: SparkContext, typ: OrderedRVDType): OrderedRVD = {
    OrderedRVD(typ,
      OrderedRVDPartitioner.empty(typ),
      sc.emptyRDD[RegionValue])
  }

  /**
    * Precondition: the iterator it is PK-sorted.  We lazily K-sort each block
    * of PK-equivalent elements.
    */
  def localKeySort(typ: OrderedRVDType,
    // it: Iterator[RegionValue[rowType]]
    it: Iterator[RegionValue]): Iterator[RegionValue] = {
    new Iterator[RegionValue] {
      private val bit = it.buffered

      private val q = new mutable.PriorityQueue[RegionValue]()(typ.kInRowOrd.reverse)

      def hasNext: Boolean = bit.hasNext || q.nonEmpty

      def next(): RegionValue = {
        if (q.isEmpty) {
          do {
            val rv = bit.next()
            // FIXME ugh, no good answer here
            q.enqueue(RegionValue(
              rv.region.copy(),
              rv.offset))
          } while (bit.hasNext && typ.pkInRowOrd.compare(q.head, bit.head) == 0)
        }

        val rv = q.dequeue()
        rv
      }
    }
  }

  // getKeys: RDD[RegionValue[kType]]
  def getKeys(typ: OrderedRVDType,
    // rdd: RDD[RegionValue[rowType]]
    rdd: RDD[RegionValue]): RDD[RegionValue] = {
    val localType = typ
    rdd.mapPartitions { it =>
      val wrv = WritableRegionValue(localType.kType)
      it.map { rv =>
        wrv.setSelect(localType.rowType, localType.kRowFieldIdx, rv)
        wrv.value
      }
    }
  }

  def getPartitionKeyInfo(typ: OrderedRVDType,
    // keys: RDD[kType]
    keys: RDD[RegionValue]): Array[OrderedRVPartitionInfo] = {
    val nPartitions = keys.getNumPartitions
    if (nPartitions == 0)
      return Array()

    val rng = new java.util.Random(1)
    val partitionSeed = Array.tabulate[Int](nPartitions)(i => rng.nextInt())

    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val localType = typ

    val pkis = keys.mapPartitionsWithIndex { case (i, it) =>
      if (it.hasNext)
        Iterator(OrderedRVPartitionInfo(localType, samplesPerPartition, i, it, partitionSeed(i)))
      else
        Iterator()
    }.collect()

    pkis.sortBy(_.min)(typ.pkOrd)
  }

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD
  ): OrderedRVD = coerce(typ, rvd, None, None)

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD,
    fastKeys: Option[RDD[RegionValue]],
    hintPartitioner: Option[OrderedRVDPartitioner]
  ): OrderedRVD = coerce(typ, rvd.rdd, fastKeys, hintPartitioner)

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue]
  ): OrderedRVD = coerce(typ, rdd, None, None)

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue],
    fastKeys: RDD[RegionValue]
  ): OrderedRVD = coerce(typ, rdd, Some(fastKeys), None)

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue],
    hintPartitioner: OrderedRVDPartitioner
  ): OrderedRVD = coerce(typ, rdd, None, Some(hintPartitioner))

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue],
    fastKeys: RDD[RegionValue],
    hintPartitioner: OrderedRVDPartitioner
  ): OrderedRVD = coerce(typ, rdd, Some(fastKeys), Some(hintPartitioner))

  def coerce(
    typ: OrderedRVDType,
    // rdd: RDD[RegionValue[rowType]]
    rdd: RDD[RegionValue],
    // fastKeys: Option[RDD[RegionValue[kType]]]
    fastKeys: Option[RDD[RegionValue]],
    hintPartitioner: Option[OrderedRVDPartitioner]
  ): OrderedRVD = {
    val sc = rdd.sparkContext

    if (rdd.partitions.isEmpty)
      return empty(sc, typ)

    // keys: RDD[RegionValue[kType]]
    val keys = fastKeys.getOrElse(getKeys(typ, rdd))

    val pkis = getPartitionKeyInfo(typ, keys)

    if (pkis.isEmpty)
      return empty(sc, typ)

    val partitionsSorted = (pkis, pkis.tail).zipped.forall { case (p, pnext) =>
      val r = typ.pkOrd.lteq(p.max, pnext.min)
      if (!r)
        log.info(s"not sorted: p = $p, pnext = $pnext")
      r
    }

    val sortedness = pkis.map(_.sortedness).min
    if (partitionsSorted && sortedness >= OrderedRVPartitionInfo.TSORTED) {
      val (adjustedPartitions, rangeBounds, adjSortedness) = rangesAndAdjustments(typ, pkis, sortedness)

      val partitioner = new OrderedRVDPartitioner(typ.partitionKey,
        typ.kType,
        rangeBounds)

      val reorderedPartitionsRDD = rdd.reorderPartitions(pkis.map(_.partitionIndex))
      val adjustedRDD = new AdjustedPartitionsRDD(reorderedPartitionsRDD, adjustedPartitions)
      (adjSortedness: @unchecked) match {
        case OrderedRVPartitionInfo.KSORTED =>
          info("Coerced sorted dataset")
          OrderedRVD(typ,
            partitioner,
            adjustedRDD)

        case OrderedRVPartitionInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          OrderedRVD(typ,
            partitioner,
            adjustedRDD.mapPartitions { it =>
              localKeySort(typ, it)
            })
      }
    } else {
      info("Ordering unsorted dataset with network shuffle")
      hintPartitioner
        .filter(_.numPartitions >= rdd.partitions.length)
        .map(adjustBoundsAndShuffle(typ, _, rdd))
        .getOrElse {
        val ranges = calculateKeyRanges(typ, pkis, rdd.getNumPartitions)
        val p = new OrderedRVDPartitioner(typ.partitionKey, typ.kType, ranges)
        shuffle(typ, p, rdd)
      }
    }
  }

  def calculateKeyRanges(typ: OrderedRVDType, pkis: Array[OrderedRVPartitionInfo], nPartitions: Int): Array[Interval] = {
    assert(nPartitions > 0)

    val pkOrd = typ.pkOrd
    var keys = pkis
      .flatMap(_.samples)
      .sorted(pkOrd)

    val min = pkis.map(_.min).min(pkOrd)
    val max = pkis.map(_.max).max(pkOrd)

    val ab = new ArrayBuilder[RegionValue]()
    ab += min
    var start = 0
    while (start < keys.length
      && pkOrd.compare(min, keys(start)) == 0)
      start += 1
    var i = 1
    while (i < nPartitions && start < keys.length) {
      var end = ((i.toDouble * keys.length) / nPartitions).toInt
      if (start > end)
        end = start
      while (end < keys.length - 1
        && pkOrd.compare(keys(end), keys(end + 1)) == 0)
        end += 1
      ab += keys(end)
      start = end + 1
      i += 1
    }
    if (pkOrd.compare(ab.last, max) != 0)
      ab += max
    val partitionEdges = ab.result()
    assert(partitionEdges.length <= nPartitions + 1)

    OrderedRVDPartitioner.makeRangeBoundIntervals(typ.pkType, partitionEdges)
  }

  def adjustBoundsAndShuffle(typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {

    val pkType = partitioner.pkType
    val pkOrdUnsafe = pkType.unsafeOrdering(true)
    val pkis = getPartitionKeyInfo(typ, OrderedRVD.getKeys(typ, rdd))

    if (pkis.isEmpty)
      return OrderedRVD(typ, partitioner, rdd)

    val min = new UnsafeRow(pkType, pkis.map(_.min).min(pkOrdUnsafe))
    val max = new UnsafeRow(pkType, pkis.map(_.max).max(pkOrdUnsafe))

    shuffle(typ, partitioner.enlargeToRange(Interval(min, max, true, true)), rdd)
  }

  def shuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rvd: RVD
  ): OrderedRVD = shuffle(typ, partitioner, rvd.rdd)

  def shuffle(typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {
    val localType = typ
    val partBc = partitioner.broadcast(rdd.sparkContext)
    OrderedRVD(typ,
      partitioner,
      new ShuffledRDD[RegionValue, RegionValue, RegionValue](
        rdd.mapPartitions { it =>
          val wrv = WritableRegionValue(localType.rowType)
          val wkrv = WritableRegionValue(localType.kType)
          it.map { rv =>
            wrv.set(rv)
            wkrv.setSelect(localType.rowType, localType.kRowFieldIdx, rv)
            (wkrv.value, wrv.value)
          }
        },
        partBc.value.sparkPartitioner(rdd.sparkContext))
        .setKeyOrdering(typ.kOrd)
        .mapPartitionsWithIndex { case (i, it) =>
          it.map { case (k, v) =>
            assert(partBc.value.getPartition(k) == i)
            v
          }
        })
  }

  def rangesAndAdjustments(typ: OrderedRVDType,
    sortedKeyInfo: Array[OrderedRVPartitionInfo],
    sortedness: Int): (IndexedSeq[Array[Adjustment[RegionValue]]], Array[Interval], Int) = {

    val partitionBounds = new ArrayBuilder[RegionValue]()
    val adjustmentsBuffer = new mutable.ArrayBuffer[Array[Adjustment[RegionValue]]]
    val indicesBuilder = new ArrayBuilder[Int]()

    var anyOverlaps = false

    val it = sortedKeyInfo.indices.iterator.buffered

    partitionBounds += sortedKeyInfo(0).min

    while (it.nonEmpty) {
      indicesBuilder.clear()
      val i = it.next()
      val thisP = sortedKeyInfo(i)
      val min = thisP.min
      val max = thisP.max

      indicesBuilder += i

      var continue = true
      while (continue && it.hasNext && typ.pkOrd.equiv(sortedKeyInfo(it.head).min, max)) {
        anyOverlaps = true
        if (typ.pkOrd.equiv(sortedKeyInfo(it.head).max, max))
          indicesBuilder += it.next()
        else {
          indicesBuilder += it.head
          continue = false
        }
      }

      val adjustments = indicesBuilder.result().zipWithIndex.map { case (partitionIndex, index) =>
        assert(sortedKeyInfo(partitionIndex).sortedness >= OrderedRVPartitionInfo.TSORTED)
        val f: (Iterator[RegionValue]) => Iterator[RegionValue] =
        // In the first partition, drop elements that should go in the last if necessary
          if (index == 0)
            if (adjustmentsBuffer.nonEmpty && typ.pkOrd.equiv(min, sortedKeyInfo(adjustmentsBuffer.last.head.index).max))
              _.dropWhile(rv => typ.pkRowOrd.compare(min.region, min.offset, rv) == 0)
            else
              identity
          else
          // In every subsequent partition, only take elements that are the max of the last
            _.takeWhile(rv => typ.pkRowOrd.compare(max.region, max.offset, rv) == 0)
        Adjustment(partitionIndex, f)
      }

      adjustmentsBuffer += adjustments
      partitionBounds += max
    }

    val pBounds = partitionBounds.result()

    val rangeBounds = OrderedRVDPartitioner.makeRangeBoundIntervals(typ.pkType, pBounds)

    val adjSortedness = if (anyOverlaps)
      sortedness.min(OrderedRVPartitionInfo.TSORTED)
    else
      sortedness

    (adjustmentsBuffer, rangeBounds, adjSortedness)
  }

  def apply(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rvd: RVD
  ): OrderedRVD = apply(typ, partitioner, rvd.rdd)

  def apply(typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {
    val sc = rdd.sparkContext

    val partitionerBc = partitioner.broadcast(sc)
    val localType = typ

    new OrderedRVD(typ, partitioner, rdd.mapPartitionsWithIndex { case (i, it) =>
      val prevK = WritableRegionValue(typ.kType)
      val prevPK = WritableRegionValue(typ.pkType)
      val pkUR = new UnsafeRow(typ.pkType)

      new Iterator[RegionValue] {
        var first = true

        def hasNext: Boolean = it.hasNext

        def next(): RegionValue = {
          val rv = it.next()

          if (first)
            first = false
          else {
            assert(localType.pkRowOrd.compare(prevPK.value, rv) <= 0 && localType.kRowOrd.compare(prevK.value, rv) <= 0)
          }

          prevK.setSelect(localType.rowType, localType.kRowFieldIdx, rv)
          prevPK.setSelect(localType.rowType, localType.pkRowFieldIdx, rv)

          pkUR.set(prevPK.value)
          assert(partitionerBc.value.rangeBounds(i).asInstanceOf[Interval].contains(localType.pkType.ordering, pkUR))

          assert(localType.pkRowOrd.compare(prevPK.value, rv) == 0)

          rv
        }
      }
    })
  }

  def union(rvds: Array[OrderedRVD]): OrderedRVD = {
    require(rvds.length > 1)
    val first = rvds(0)
    val sc = first.sparkContext
    OrderedRVD.coerce(
      first.typ,
      sc.union(rvds.map(_.rdd)),
      None,
      None)
  }
}
