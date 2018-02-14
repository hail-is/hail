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

class OrderedRVD private(
  val typ: OrderedRVDType,
  val partitioner: OrderedRVDPartitioner,
  val rdd: RDD[RegionValue]) extends RVD with Serializable {
  self =>
  def rowType: TStruct = typ.rowType

  // should be totally generic, permitting any number of keys, but that requires more work
  def downcastToPK(): OrderedRVD = {
    val newType = new OrderedRVDType(partitionKey = typ.partitionKey,
      key = typ.partitionKey,
      rowType = rowType)
    OrderedRVD(newType, partitioner, rdd)
  }

  def upcast(castKeys: Array[String]): OrderedRVD = {
    val newType = new OrderedRVDType(partitionKey = typ.partitionKey,
      key = typ.key ++ castKeys,
      rowType = rowType)
    OrderedRVD(newType, partitioner, rdd)
  }

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

  def zipPartitionsPreservesPartitioning[T](newTyp: OrderedRVDType, rdd2: RDD[T])(f: (Iterator[RegionValue], Iterator[T]) => Iterator[RegionValue])(implicit tct: ClassTag[T]): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      rdd.zipPartitions(rdd2) { case (it, it2) =>
        f(it, it2)
      })

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

  def orderedJoinDistinct(right: OrderedRVD, joinType: String): RDD[JoinedRegionValue] = {
    val lTyp = typ
    val rTyp = right.typ

    if (!lTyp.kType.fieldType.sameElements(rTyp.kType.fieldType))
      fatal(
        s"""Incompatible join keys.  Keys must have same length and types, in order:
           | Left key type: ${ lTyp.kType.toString }
           | Right key type: ${ rTyp.kType.toString }
         """.stripMargin)

    joinType match {
      case "inner" | "left" => new OrderedJoinDistinctRDD2(this, right, joinType)
      case _ => fatal(s"Unknown join type `$joinType'. Choose from `inner' or `left'.")
    }
  }

  def orderedZipJoin(right: OrderedRVD): OrderedZipJoinRDD = {
    val pkOrd = this.partitioner.pkType.ordering
    if (pkOrd.gt(this.partitioner.minBound, right.partitioner.minBound) ||
      pkOrd.lt(this.partitioner.maxBound, right.partitioner.maxBound)) {

      val newRangeBounds = partitioner.rangeBounds.toArray

      newRangeBounds(0) = newRangeBounds(0).asInstanceOf[Interval]
        .copy(start = pkOrd.min(this.partitioner.minBound, right.partitioner.minBound))
      newRangeBounds(newRangeBounds.length - 1) = newRangeBounds(newRangeBounds.length - 1).asInstanceOf[Interval]
        .copy(end = pkOrd.max(this.partitioner.maxBound, right.partitioner.maxBound))

      val newPartitioner = new OrderedRVDPartitioner(partitioner.partitionKey,
        partitioner.kType, UnsafeIndexedSeq(partitioner.rangeBoundsType, newRangeBounds))

      new OrderedZipJoinRDD(OrderedRVD(this.typ, newPartitioner, this.rdd), right)

    } else {
      new OrderedZipJoinRDD(this, right)
    }
  }

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
      if (!partitioner.rangeTree.overlaps(pkOrdering, i))
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
      val newRDD = new AdjustedPartitionsRDD(rdd, newPartitionIndices.map(i => Array(Adjustment(i, (it: Iterator[RegionValue]) => it.filter(pred)))))
      OrderedRVD(typ,
        partitioner.copy(numPartitions = newPartitionIndices.length,
          rangeBounds = UnsafeIndexedSeq(partitioner.rangeBoundsType, newPartitionIndices.map(partitioner.rangeBounds))),
        newRDD)
    }
  }

  def head(n: Long): OrderedRVD = {
    require(n >= 0)

    if (n == 0)
      return OrderedRVD.empty(sparkContext, typ)

    val newRDD = rdd.head(n)
    val newNParts = newRDD.getNumPartitions
    assert(newNParts > 0)

    val newRangeBounds = (0 until newNParts).map(partitioner.rangeBounds)
    val newPartitioner = new OrderedRVDPartitioner(partitioner.partitionKey,
      partitioner.kType,
      UnsafeIndexedSeq(partitioner.rangeBoundsType, newRangeBounds))

    OrderedRVD(typ, newPartitioner, newRDD)
  }

  def groupByKey(valuesField: String = "values"): OrderedRVD = {
    val newTyp = new OrderedRVDType(
      typ.partitionKey,
      typ.key,
      typ.kType ++ TStruct(valuesField -> TArray(typ.valueType)))
    val newRowType = newTyp.rowType

    val newRDD: RDD[RegionValue] = rdd.mapPartitions { it =>
      new Iterator[RegionValue] {
        val wrv = WritableRegionValue(typ.kType)

        var peekRV: RegionValue =
          if (it.hasNext)
            it.next()
          else
            null

        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv2 = RegionValue(region)

        val ab = new ArrayBuilder[Long]()

        var present: Boolean = false

        def advance() {
          region.clear()
          assert(ab.isEmpty)

          wrv.setSelect(typ.rowType, typ.kRowFieldIdx, peekRV)
          do {
            rvb.start(typ.valueType)
            rvb.startStruct()
            rvb.addFields(typ.rowType, peekRV, typ.valueFieldIdx)
            rvb.endStruct()
            ab += rvb.end()
            peekRV = if (it.hasNext) it.next() else null
          } while (peekRV != null
            && typ.kRowOrd.compare(wrv.region, wrv.offset, peekRV) == 0)

          rvb.start(newRowType)
          rvb.startStruct()
          var i = 0
          while (i < typ.kType.size) {
            rvb.addField(typ.kType, wrv.value, i)
            i += 1
          }
          rvb.startArray(ab.length)
          i = 0
          while (i < ab.length) {
            rvb.addRegionValue(typ.valueType, region, ab(i))
            i += 1
          }
          ab.clear()
          rvb.endArray()
          rvb.endStruct()
          rv2.setOffset(rvb.end())

          present = true
        }

        def hasNext: Boolean = {
          if (!present && peekRV != null)
            advance()
          present
        }

        def next(): RegionValue = {
          if (!hasNext)
            throw new NoSuchElementException("next on empty iterator")
          present = false
          rv2
        }
      }
    }

    OrderedRVD(newTyp, partitioner, newRDD)
  }

  def subsetPartitions(keep: Array[Int]): OrderedRVD = {
    require(keep.length <= rdd.partitions.length, "tried to subset to more partitions than exist")
    require(keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < rdd.partitions.length)),
      "values not sorted or not in range [0, number of partitions)")

    val newRangeBounds = keep.map(partitioner.rangeBounds)
    val newPartitioner = new OrderedRVDPartitioner(
      partitioner.partitionKey,
      partitioner.kType,
      UnsafeIndexedSeq(partitioner.rangeBoundsType, newRangeBounds))

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
}

object OrderedRVD {
  type CoercionMethod = Int

  final val ORDERED_PARTITIONER: CoercionMethod = 0
  final val AS_IS: CoercionMethod = 1
  final val LOCAL_SORT: CoercionMethod = 2
  final val SHUFFLE: CoercionMethod = 3

  def empty(sc: SparkContext, typ: OrderedRVDType): OrderedRVD = {
    OrderedRVD(typ,
      OrderedRVDPartitioner.empty(typ),
      sc.emptyRDD[RegionValue])
  }

  def cast(typ: OrderedRVDType,
    rdd: RDD[RegionValue]): OrderedRVD = {
    if (rdd.partitions.isEmpty)
      OrderedRVD.empty(rdd.sparkContext, typ)
    else
      (rdd.partitioner: @unchecked) match {
        case Some(p: OrderedRVDPartitioner) => OrderedRVD(typ, p.asInstanceOf[OrderedRVDPartitioner], rdd)
      }
  }

  def apply(typ: OrderedRVDType,
    rdd: RDD[RegionValue], fastKeys: Option[RDD[RegionValue]], hintPartitioner: Option[OrderedRVDPartitioner]): OrderedRVD = {
    val (_, orderedRDD) = coerce(typ, rdd, fastKeys, hintPartitioner)
    orderedRDD
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
    rdd.mapPartitions { it =>
      val wrv = WritableRegionValue(typ.kType)
      it.map { rv =>
        wrv.setSelect(typ.rowType, typ.kRowFieldIdx, rv)
        wrv.value
      }
    }
  }

  def getPartitionKeyInfo(typ: OrderedRVDType,
    // keys: RDD[kType]
    keys: RDD[RegionValue]): Array[OrderedRVPartitionInfo] = {
    val nPartitions = keys.getNumPartitions

    val rng = new java.util.Random(1)
    val partitionSeed = Array.tabulate[Int](nPartitions)(i => rng.nextInt())

    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val pkis = keys.mapPartitionsWithIndex { case (i, it) =>
      if (it.hasNext)
        Iterator(OrderedRVPartitionInfo(typ, samplesPerPartition, i, it, partitionSeed(i)))
      else
        Iterator()
    }.collect()

    pkis.sortBy(_.min)(typ.pkOrd)
  }

  def coerce(typ: OrderedRVDType,
    // rdd: RDD[RegionValue[rowType]]
    rdd: RDD[RegionValue],
    // fastKeys: Option[RDD[RegionValue[kType]]]
    fastKeys: Option[RDD[RegionValue]] = None,
    hintPartitioner: Option[OrderedRVDPartitioner] = None): (CoercionMethod, OrderedRVD) = {
    val sc = rdd.sparkContext

    if (rdd.partitions.isEmpty)
      return (ORDERED_PARTITIONER, empty(sc, typ))

    // keys: RDD[RegionValue[kType]]
    val keys = fastKeys.getOrElse(getKeys(typ, rdd))

    val pkis = getPartitionKeyInfo(typ, keys)

    if (pkis.isEmpty)
      return (AS_IS, empty(sc, typ))

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
          (AS_IS, OrderedRVD(typ,
            partitioner,
            adjustedRDD))

        case OrderedRVPartitionInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          (LOCAL_SORT, OrderedRVD(typ,
            partitioner,
            adjustedRDD.mapPartitions { it =>
              localKeySort(typ, it)
            }))
      }
    } else {
      info("Ordering unsorted dataset with network shuffle")
      val p = hintPartitioner
        .filter(_.numPartitions >= rdd.partitions.length)
        .getOrElse {
          val ranges = calculateKeyRanges(typ, pkis, rdd.getNumPartitions)
          new OrderedRVDPartitioner(typ.partitionKey, typ.kType, ranges)
        }
      (SHUFFLE, shuffle(typ, p, rdd))
    }
  }

  def calculateKeyRanges(typ: OrderedRVDType, pkis: Array[OrderedRVPartitionInfo], nPartitions: Int): UnsafeIndexedSeq = {
    val pkOrd = typ.pkOrd
    var keys = pkis
      .flatMap(_.samples)
      .sorted(pkOrd)

    val min = pkis.map(_.min).min(pkOrd)
    val max = pkis.map(_.max).max(pkOrd)

    keys = min +: keys :+ max

    val ab = new ArrayBuilder[RegionValue]()
    var i = 0
    while (i < keys.length) {
      if (i == 0
        || pkOrd.compare(keys(i - 1), keys(i)) != 0)
        ab += keys(i)
      i += 1
    }
    keys = ab.result()

    // FIXME weighted
    val partitionMaxes =
      if (keys.length <= nPartitions + 1)
        keys
      else {
        val k = keys.length / nPartitions
        assert(k > 0)
        Array.tabulate(nPartitions)(i => keys(i * k)) :+ max
      }

    OrderedRVDPartitioner.makeRangeBoundIntervals(typ.pkType, partitionMaxes)
  }

  def adjustBoundsAndShuffle(typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {

    val pkType = partitioner.pkType
    val pkOrdUnsafe = pkType.unsafeOrdering(true)
    val pkOrd = pkType.ordering.toOrdering
    val pkis = getPartitionKeyInfo(typ, OrderedRVD.getKeys(typ, rdd))

    if (pkis.isEmpty)
      return OrderedRVD(typ, partitioner, rdd)

    val min = new UnsafeRow(pkType, pkis.map(_.min).min(pkOrdUnsafe))
    val max = new UnsafeRow(pkType, pkis.map(_.max).max(pkOrdUnsafe))

    val newRangeBounds = partitioner.rangeBounds.toArray

    newRangeBounds(0) = newRangeBounds(0).asInstanceOf[Interval]
      .copy(start = pkOrd.min(newRangeBounds(0).asInstanceOf[Interval].start, min))

    newRangeBounds(newRangeBounds.length - 1) = newRangeBounds(newRangeBounds.length - 1).asInstanceOf[Interval]
      .copy(end = pkOrd.max(newRangeBounds(newRangeBounds.length - 1).asInstanceOf[Interval].end, max))

    val newPartitioner = new OrderedRVDPartitioner(partitioner.partitionKey,
      partitioner.kType, UnsafeIndexedSeq(partitioner.rangeBoundsType, newRangeBounds))

    shuffle(typ, newPartitioner, rdd)

  }

  def shuffle(typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {
    OrderedRVD(typ,
      partitioner,
      new ShuffledRDD[RegionValue, RegionValue, RegionValue](
        rdd.mapPartitions { it =>
          val wkrv = WritableRegionValue(typ.kType)
          it.map { rv =>
            wkrv.setSelect(typ.rowType, typ.kRowFieldIdx, rv)
            (wkrv.value, rv)
          }
        },
        partitioner)
        .setKeyOrdering(typ.kOrd)
        .mapPartitionsWithIndex { case (i, it) =>
          it.map { case (k, v) =>
            assert(partitioner.getPartition(k) == i)
            v
          }
        })
  }

  def shuffle(typ: OrderedRVDType, partitioner: OrderedRVDPartitioner, rvd: RVD): OrderedRVD =
    shuffle(typ, partitioner, rvd.rdd)

  def rangesAndAdjustments(typ: OrderedRVDType,
    sortedKeyInfo: Array[OrderedRVPartitionInfo],
    sortedness: Int): (IndexedSeq[Array[Adjustment[RegionValue]]], UnsafeIndexedSeq, Int) = {

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

  def apply(typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {
    val sc = rdd.sparkContext

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
            assert(typ.pkRowOrd.compare(prevPK.value, rv) <= 0 && typ.kRowOrd.compare(prevK.value, rv) <= 0)
          }

          prevK.setSelect(typ.rowType, typ.kRowFieldIdx, rv)
          prevPK.setSelect(typ.rowType, typ.pkRowFieldIdx, rv)

          pkUR.set(prevPK.value)
          assert(partitioner.rangeBounds(i).asInstanceOf[Interval].contains(typ.pkType.ordering, pkUR))

          assert(typ.pkRowOrd.compare(prevPK.value, rv) == 0)

          rv
        }
      }
    })
  }

  def apply(typ: OrderedRVDType, partitioner: OrderedRVDPartitioner, rvd: RVD): OrderedRVD = {
    assert(typ.rowType == rvd.rowType)
    apply(typ, partitioner, rvd.rdd)
  }
}
