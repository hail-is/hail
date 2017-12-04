package is.hail.rvd

import java.util

import is.hail.annotations._
import is.hail.expr.{TArray, Type}
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext

import scala.collection.mutable

class OrderedRVD private(
  val typ: OrderedRVType,
  val partitioner: OrderedRVPartitioner,
  val rdd: RDD[RegionValue]) extends RVD { self =>
  def rowType: Type = typ.rowType

  def insert[PC](newContext: () => PC)(typeToInsert: Type,
    path: List[String],
    // rv argument to add is the entire row
    add: (PC, RegionValue, RegionValueBuilder) => Unit): OrderedRVD = {
    val localTyp = typ

    val (insTyp, inserter) = typ.insert(typeToInsert, path)
    OrderedRVD(insTyp,
      partitioner,
      rdd.mapPartitions { it =>
        val c = newContext()
        val rv2b = new RegionValueBuilder()
        val rv2 = RegionValue()

        it.map { rv =>
          val ur = new UnsafeRow(localTyp.rowType, rv)
          rv2b.set(rv.region)
          rv2b.start(insTyp.rowType)
          inserter(rv.region, rv.offset, rv2b, () => add(c, rv, rv2b))
          rv2.set(rv.region, rv2b.end())
          rv2
        }
      })
  }

  def mapPreservesPartitioning(newTyp: OrderedRVType)(f: (RegionValue) => RegionValue): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      rdd.map(f))

  def mapPartitionsPreservesPartitioning(newTyp: OrderedRVType)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      rdd.mapPartitions(f))

  override def filter(p: (RegionValue) => Boolean): OrderedRVD =
    OrderedRVD(typ,
      partitioner,
      rdd.filter(p))

  def sample(withReplacement: Boolean, fraction: Double, seed: Long): OrderedRVD =
    OrderedRVD(typ,
      partitioner,
      rdd.sample(withReplacement, fraction, seed))

  override def persist(level: StorageLevel): OrderedRVD = {
    val PersistedRVRDD(persistedRDD, iterationRDD) = persistRVRDD(level)
    new OrderedRVD(typ, partitioner, iterationRDD) {
      override def storageLevel: StorageLevel = level

      override def persist(level: StorageLevel): OrderedRVD = throw new IllegalArgumentException("already persisted")

      override def unpersist(): OrderedRVD = {
        persistedRDD.unpersist()
        self
      }
    }
  }

  override def unpersist(): OrderedRVD = throw new IllegalArgumentException("not persisted")

  def orderedJoinDistinct(right: OrderedRVD, joinType: String): RDD[JoinedRegionValue] = {
    val lTyp = typ
    val rTyp = right.typ

    if (lTyp.kType != rTyp.kType)
      fatal(
        s"""Incompatible join keys.  Keys must have same length and types, in order:
           | Left key type: ${ lTyp.kType.toPrettyString(compact = true) }
           | Right key type: ${ rTyp.kType.toPrettyString(compact = true) }
         """.stripMargin)

    joinType match {
      case "inner" | "left" => new OrderedJoinDistinctRDD2(this, right, joinType)
      case _ => fatal(s"Unknown join type `$joinType'. Choose from `inner' or `left'.")
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

  def copy(typ: OrderedRVType = typ,
    orderedPartitioner: OrderedRVPartitioner = partitioner,
    rdd: RDD[RegionValue] = rdd): OrderedRVD = {
    OrderedRVD(typ, orderedPartitioner, rdd)
  }

  def naiveCoalesce(maxPartitions: Int): OrderedRVD = {
    val n = partitioner.numPartitions
    if (maxPartitions >= n)
      return this

    val newN = maxPartitions
    val newNParts = Array.tabulate(newN)(i => (n - i + newN - 1) / newN)
    assert(newNParts.sum == n)
    assert(newNParts.forall(_ > 0))

    val newPartEnd = newNParts.scanLeft(-1)(_ + _).tail
    assert(newPartEnd.last == n - 1)

    val newRangeBounds = UnsafeIndexedSeq(
      TArray(typ.pkType),
      newPartEnd.init.map(partitioner.rangeBounds))

    OrderedRVD(
      typ,
      new OrderedRVPartitioner(newN, typ.partitionKey, typ.kType, newRangeBounds),
      new BlockedRDD(rdd, newPartEnd))
  }

  override def coalesce(maxPartitions: Int, shuffle: Boolean): OrderedRVD = {
    require(maxPartitions > 0, "cannot coalesce to nPartitions <= 0")
    val n = rdd.partitions.length
    if (!shuffle && maxPartitions >= n)
      return this
    if (shuffle) {
      val shuffled = rdd.coalesce(maxPartitions, shuffle = true)
      val ranges = OrderedRVD.calculateKeyRanges(typ, OrderedRVD.getPartitionKeyInfo(typ, shuffled), shuffled.getNumPartitions)
      OrderedRVD.shuffle(typ, new OrderedRVPartitioner(ranges.length + 1, typ.partitionKey, typ.kType, ranges), shuffled)
    } else {

      val partSize = rdd.context.runJob(rdd, getIteratorSize _)
      log.info(s"partSize = ${ partSize.toSeq }")

      val partCumulativeSize = mapAccumulate[Array, Long, Long, Long](partSize, 0)((s, acc) => (s + acc, s + acc))
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

      info(s"newPartEnd = ${ newPartEnd.toSeq }")

      assert(newPartEnd.last == n - 1)
      assert(newPartEnd.zip(newPartEnd.tail).forall { case (i, inext) => i < inext })

      if (newPartEnd.length < maxPartitions)
        warn(s"coalesced to ${ newPartEnd.length } ${ plural(newPartEnd.length, "partition") }, less than requested $maxPartitions")

      val newRangeBounds = newPartEnd.init.map(partitioner.rangeBounds).asInstanceOf[UnsafeIndexedSeq]
      val newPartitioner = new OrderedRVPartitioner(newRangeBounds.length + 1, typ.partitionKey, typ.kType, newRangeBounds)
      OrderedRVD(typ, newPartitioner, new BlockedRDD(rdd, newPartEnd))
    }
  }
}

object OrderedRVD {
  type CoercionMethod = Int

  final val ORDERED_PARTITIONER: CoercionMethod = 0
  final val AS_IS: CoercionMethod = 1
  final val LOCAL_SORT: CoercionMethod = 2
  final val SHUFFLE: CoercionMethod = 3

  def empty(sc: SparkContext, typ: OrderedRVType): OrderedRVD = {
    OrderedRVD(typ,
      OrderedRVPartitioner.empty(typ),
      sc.emptyRDD[RegionValue])
  }

  def cast(typ: OrderedRVType,
    rdd: RDD[RegionValue]): OrderedRVD = {
    if (rdd.partitions.isEmpty)
      OrderedRVD.empty(rdd.sparkContext, typ)
    else
      (rdd.partitioner: @unchecked) match {
        case Some(p: OrderedRVPartitioner) => OrderedRVD(typ, p.asInstanceOf[OrderedRVPartitioner], rdd)
      }
  }

  def apply(typ: OrderedRVType,
    rdd: RDD[RegionValue], fastKeys: Option[RDD[RegionValue]], hintPartitioner: Option[OrderedRVPartitioner]): OrderedRVD = {
    val (_, orderedRDD) = coerce(typ, rdd, fastKeys, hintPartitioner)
    orderedRDD
  }

  /**
    * Precondition: the iterator it is PK-sorted.  We lazily K-sort each block
    * of PK-equivalent elements.
    */
  def localKeySort(typ: OrderedRVType,
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
  def getKeys(typ: OrderedRVType,
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

  def getPartitionKeyInfo(typ: OrderedRVType,
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

  def coerce(typ: OrderedRVType,
    // rdd: RDD[RegionValue[rowType]]
    rdd: RDD[RegionValue],
    // fastKeys: Option[RDD[RegionValue[kType]]]
    fastKeys: Option[RDD[RegionValue]] = None,
    hintPartitioner: Option[OrderedRVPartitioner] = None): (CoercionMethod, OrderedRVD) = {
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
    if (partitionsSorted && sortedness >= PartitionKeyInfo.TSORTED) {
      val (adjustedPartitions, rangeBounds, adjSortedness) = rangesAndAdjustments(typ, pkis, sortedness)

      val unsafeRangeBounds = UnsafeIndexedSeq(TArray(typ.pkType), rangeBounds)
      val partitioner = new OrderedRVPartitioner(adjustedPartitions.length,
        typ.partitionKey,
        typ.kType,
        unsafeRangeBounds)

      val reorderedPartitionsRDD = rdd.reorderPartitions(pkis.map(_.partitionIndex))
      val adjustedRDD = new AdjustedPartitionsRDD(reorderedPartitionsRDD, adjustedPartitions)
      (adjSortedness: @unchecked) match {
        case PartitionKeyInfo.KSORTED =>
          info("Coerced sorted dataset")
          (AS_IS, OrderedRVD(typ,
            partitioner,
            adjustedRDD))

        case PartitionKeyInfo.TSORTED =>
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
          new OrderedRVPartitioner(ranges.length + 1, typ.partitionKey, typ.kType, ranges)
        }
      (SHUFFLE, shuffle(typ, p, rdd))
    }
  }

  def calculateKeyRanges(typ: OrderedRVType, pkis: Array[OrderedRVPartitionInfo], nPartitions: Int): UnsafeIndexedSeq = {
    val keys = pkis
      .flatMap(_.samples)
      .sorted(typ.pkOrd)

    // FIXME weighted
    val rangeBounds =
      if (keys.length <= nPartitions)
        keys.init
      else {
        val k = keys.length / nPartitions
        assert(k > 0)
        Array.tabulate(nPartitions - 1)(i => keys((i + 1) * k))
      }

    UnsafeIndexedSeq(TArray(typ.pkType), rangeBounds)
  }

  def shuffle(typ: OrderedRVType,
    partitioner: OrderedRVPartitioner,
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

  def shuffle(typ: OrderedRVType, partitioner: OrderedRVPartitioner, rvd: RVD): OrderedRVD =
    shuffle(typ, partitioner, rvd.rdd)

  def rangesAndAdjustments(typ: OrderedRVType,
    sortedKeyInfo: Array[OrderedRVPartitionInfo],
    sortedness: Int): (IndexedSeq[Array[Adjustment[RegionValue]]], Array[RegionValue], Int) = {

    val rangeBounds = new ArrayBuilder[RegionValue]()
    val adjustmentsBuffer = new mutable.ArrayBuffer[Array[Adjustment[RegionValue]]]
    val indicesBuilder = new ArrayBuilder[Int]()

    var anyOverlaps = false

    val it = sortedKeyInfo.indices.iterator.buffered

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
        assert(sortedKeyInfo(partitionIndex).sortedness >= PartitionKeyInfo.TSORTED)
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

      if (it.hasNext)
        rangeBounds += max
    }

    val adjSortedness = if (anyOverlaps)
      sortedness.min(PartitionKeyInfo.TSORTED)
    else
      sortedness

    (adjustmentsBuffer, rangeBounds.result(), adjSortedness)
  }

  def apply(typ: OrderedRVType,
    partitioner: OrderedRVPartitioner,
    rdd: RDD[RegionValue]): OrderedRVD = {
    val sc = rdd.sparkContext

    new OrderedRVD(typ, partitioner, rdd.mapPartitionsWithIndex { case (i, it) =>
      val prev = WritableRegionValue(typ.pkType)

      new Iterator[RegionValue] {
        var first = true

        def hasNext: Boolean = it.hasNext

        def next(): RegionValue = {
          val rv = it.next()

          if (i < partitioner.rangeBounds.length) {
            assert(typ.pkRowOrd.compare(
              partitioner.region, partitioner.loadElement(i),
              rv) >= 0)
          }
          if (i > 0)
            assert(typ.pkRowOrd.compare(partitioner.region, partitioner.loadElement(i - 1),
              rv) < 0)

          if (first)
            first = false
          else
            assert(typ.pkRowOrd.compare(prev.value, rv) <= 0)

          prev.setSelect(typ.rowType, typ.pkRowFieldIdx, rv)

          assert(typ.pkRowOrd.compare(prev.value, rv) == 0)

          rv
        }
      }
    })
  }

  def apply(typ: OrderedRVType, partitioner: OrderedRVPartitioner, rvd: RVD): OrderedRVD =
    apply(typ, partitioner, rvd.rdd)
}
