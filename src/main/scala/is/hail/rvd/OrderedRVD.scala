package is.hail.rvd

import java.util

import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.PruneDeadFields.isSupertype
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.language.existentials
import scala.reflect.ClassTag

abstract class RVDCoercer(val fullType: OrderedRVDType) {
  final def coerce(typ: OrderedRVDType, crdd: ContextRDD[RVDContext, RegionValue]): OrderedRVD = {
    require(isSupertype(typ.rowType, fullType.rowType))
    require(typ.key.sameElements(fullType.key))
    require(typ.partitionKey.sameElements(fullType.partitionKey))
    _coerce(typ, crdd)
  }

  protected def _coerce(typ: OrderedRVDType, crdd: ContextRDD[RVDContext, RegionValue]): OrderedRVD
}

class OrderedRVD(
  val typ: OrderedRVDType,
  val partitioner: OrderedRVDPartitioner,
  val crdd: ContextRDD[RVDContext, RegionValue]
) extends RVD {
  self =>
  require(crdd.getNumPartitions == partitioner.numPartitions)

  def boundary: OrderedRVD = OrderedRVD(typ, partitioner, crddBoundary)

  def rowType: TStruct = typ.rowType

  def updateType(newTyp: OrderedRVDType): OrderedRVD =
    OrderedRVD(newTyp, partitioner, crdd)

  def mapPreservesPartitioning(newTyp: OrderedRVDType)(f: (RegionValue) => RegionValue): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      crdd.map(f))

  def mapPartitionsWithIndexPreservesPartitioning(newTyp: OrderedRVDType)(f: (Int, Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      crdd.mapPartitionsWithIndex(f))

  def mapPartitionsWithIndexPreservesPartitioning(
    newTyp: OrderedRVDType,
    f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = OrderedRVD(
    newTyp,
    partitioner,
    crdd.cmapPartitionsWithIndex(f))

  def mapPartitionsPreservesPartitioning(newTyp: OrderedRVDType)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRVD =
    OrderedRVD(newTyp,
      partitioner,
      crdd.mapPartitions(f))

  def mapPartitionsPreservesPartitioning(
    newTyp: OrderedRVDType,
    f: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = OrderedRVD(newTyp, partitioner, crdd.cmapPartitions(f))

  override def filter(p: (RegionValue) => Boolean): OrderedRVD =
    OrderedRVD(typ, partitioner, crddBoundary.filter(p))

  def filterWithContext[C](makeContext: RVDContext => C, f: (C, RegionValue) => Boolean): RVD = {
    mapPartitionsPreservesPartitioning(typ, { (context, it) =>
      val c = makeContext(context)
      it.filter { rv =>
        if (f(c, rv))
          true
        else {
          rv.region.clear()
          false
        }
      }
    })
  }

  def sample(withReplacement: Boolean, p: Double, seed: Long): OrderedRVD =
    OrderedRVD(typ, partitioner, crdd.sample(withReplacement, p, seed))

  def zipWithIndex(name: String): OrderedRVD = {
    assert(!typ.key.contains(name))
    val (newRowType, newCRDD) = zipWithIndexCRDD(name)

    OrderedRVD(
      typ.copy(rowType = newRowType.asInstanceOf[TStruct]),
      partitioner,
      crdd = newCRDD
    )
  }

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

    new OrderedRVD(
      typ = ordType,
      partitioner = newPartitioner,
      crdd = ContextRDD(RepartitionedOrderedRDD(this, newPartitioner)))
  }

  def keyBy(key: Array[String] = typ.key): KeyedOrderedRVD =
    new KeyedOrderedRVD(this, key)

  def orderedJoin(
    right: OrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD =
    keyBy().orderedJoin(right.keyBy(), joinType, joiner, joinedType)

  def orderedJoinDistinct(
    right: OrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD =
    keyBy().orderedJoinDistinct(right.keyBy(), joinType, joiner, joinedType)

  def orderedZipJoin(right: OrderedRVD): ContextRDD[RVDContext, JoinedRegionValue] =
    keyBy().orderedZipJoin(right.keyBy())

  def partitionSortedUnion(rdd2: OrderedRVD): OrderedRVD = {
    assert(typ == rdd2.typ)
    assert(partitioner == rdd2.partitioner)

    val localTyp = typ
    zipPartitions(typ, partitioner, rdd2) { (ctx, it, it2) =>
      new Iterator[RegionValue] {
        private val bit = it.buffered
        private val bit2 = it2.buffered
        private val rv = RegionValue()

        def hasNext: Boolean = bit.hasNext || bit2.hasNext

        def next(): RegionValue = {
          val old =
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
          ctx.rvb.start(localTyp.rowType)
          ctx.rvb.addRegionValue(localTyp.rowType, old)
          rv.set(ctx.region, ctx.rvb.end())
          rv
        }
      }
    }
  }

  def copy(typ: OrderedRVDType = typ,
    orderedPartitioner: OrderedRVDPartitioner = partitioner,
    rdd: ContextRDD[RVDContext, RegionValue] = crdd): OrderedRVD = {
    OrderedRVD(typ, orderedPartitioner, rdd)
  }

  def blockCoalesce(partitionEnds: Array[Int]): OrderedRVD = {
    assert(partitionEnds.last == partitioner.numPartitions - 1 && partitionEnds(0) >= 0)
    assert(partitionEnds.zip(partitionEnds.tail).forall { case (i, inext) => i < inext })
    OrderedRVD(typ, partitioner.coalesceRangeBounds(partitionEnds), crdd.blocked(partitionEnds))
  }

  def naiveCoalesce(maxPartitions: Int): OrderedRVD = {
    val n = partitioner.numPartitions
    if (maxPartitions >= n)
      return this

    val newN = maxPartitions
    val newNParts = partition(n, newN)
    assert(newNParts.forall(_ > 0))
    blockCoalesce(newNParts.scanLeft(-1)(_ + _).tail)
  }

  override def coalesce(maxPartitions: Int, shuffle: Boolean): OrderedRVD = {
    require(maxPartitions > 0, "cannot coalesce to nPartitions <= 0")
    val n = crdd.partitions.length
    if (!shuffle && maxPartitions >= n)
      return this
    if (shuffle) {
      val shuffled = stably(_.shuffleCoalesce(maxPartitions))
      val pki = OrderedRVD.getPartitionKeyInfo(typ, OrderedRVD.getKeys(typ, shuffled))
      if (pki.isEmpty)
        return OrderedRVD.empty(sparkContext, typ)
      val ranges = OrderedRVD.calculateKeyRanges(
        typ,
        pki,
        shuffled.getNumPartitions)
      OrderedRVD.shuffle(
        typ,
        new OrderedRVDPartitioner(typ.partitionKey, typ.kType, ranges),
        shuffled)
    } else {

      val partSize = countPerPartition()
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

  def filterIntervals(intervals: IntervalTree[_], keep: Boolean): OrderedRVD = {
    if (keep)
      filterToIntervals(intervals)
    else
      filterOutIntervals(intervals)
  }

  def filterOutIntervals(intervals: IntervalTree[_]): OrderedRVD = {
    val intervalsBc = crdd.sparkContext.broadcast(intervals)
    val pkType = typ.pkType
    val pkRowFieldIdx = typ.pkRowFieldIdx
    val rowType = typ.rowType

    mapPartitionsPreservesPartitioning(typ, { (ctx, it) =>
      val pkUR = new UnsafeRow(pkType)
      it.filter { rv =>
        ctx.rvb.start(pkType)
        ctx.rvb.selectRegionValue(rowType, pkRowFieldIdx, rv)
        pkUR.set(ctx.region, ctx.rvb.end())
        !intervalsBc.value.contains(pkType.ordering, pkUR)
      }
    })
  }

  def filterToIntervals(intervals: IntervalTree[_]): OrderedRVD = {
    val pkOrdering = typ.pkType.ordering
    val intervalsBc = crdd.sparkContext.broadcast(intervals)
    val rowType = typ.rowType
    val pkRowFieldIdx = typ.pkRowFieldIdx

    val pred: (RegionValue) => Boolean = (rv: RegionValue) => {
      val ur = new UnsafeRow(rowType, rv)
      val pk = Row.fromSeq(
        pkRowFieldIdx.map(i => ur.get(i)))
      intervalsBc.value.contains(pkOrdering, pk)
    }

    val nPartitions = getNumPartitions
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
      .toSet[Int] // distinct
      .toArray
      .sorted

    info(s"interval filter loaded ${ newPartitionIndices.length } of $nPartitions partitions")

    if (newPartitionIndices.isEmpty)
      OrderedRVD.empty(sparkContext, typ)
    else {
      subsetPartitions(newPartitionIndices).filter(pred)
    }
  }

  def head(n: Long): OrderedRVD = {
    require(n >= 0)

    if (n == 0)
      return OrderedRVD.empty(sparkContext, typ)

    val newRDD = crdd.head(n)
    val newNParts = newRDD.getNumPartitions
    assert(newNParts >= 0)

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

    OrderedRVD(newTyp, partitioner, crdd.cmapPartitionsAndContext { (consumerCtx, useCtxes) =>
      val consumerRegion = consumerCtx.region
      val rvb = consumerCtx.rvb
      val outRV = RegionValue(consumerRegion)

      val bufferRegion = consumerCtx.freshContext.region
      val buffer = new RegionValueArrayBuffer(localType.valueType, bufferRegion)

      val producerCtx = consumerCtx.freshContext
      val producerRegion = producerCtx.region
      val it = useCtxes.flatMap(_ (producerCtx))

      val stepped = OrderedRVIterator(
        localType,
        it,
        consumerCtx.freshContext
      ).staircase

      stepped.map { stepIt =>
        buffer.clear()
        rvb.start(newRowType)
        rvb.startStruct()
        var i = 0
        while (i < localType.kType.size) {
          rvb.addField(localType.rowType, stepIt.value, localType.kRowFieldIdx(i))
          i += 1
        }
        for (rv <- stepIt) {
          buffer.appendSelect(localType.rowType, localType.valueFieldIdx, rv)
          producerRegion.clear()
        }
        rvb.startArray(buffer.length)
        for (rv <- buffer)
          rvb.addRegionValue(localType.valueType, rv)
        rvb.endArray()
        rvb.endStruct()
        outRV.setOffset(rvb.end())
        outRV
      }
    })
  }

  def distinctByKey(): OrderedRVD = {
    val localType = typ
    mapPartitionsPreservesPartitioning(typ, (ctx, it) =>
      OrderedRVIterator(localType, it, ctx)
        .staircase
        .map(_.value)
    )
  }

  def subsetPartitions(keep: Array[Int]): OrderedRVD = {
    require(keep.length <= crdd.partitions.length, "tried to subset to more partitions than exist")
    require(keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < crdd.partitions.length)),
      "values not increasing or not in range [0, number of partitions)")

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

    OrderedRVD(typ, newPartitioner, crdd.subsetPartitions(keep))
  }

  override protected def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]): RVDSpec =
    OrderedRVDSpec(
      typ,
      codecSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(
        partitioner.rangeBounds,
        partitioner.rangeBoundsType))

  def zipPartitionsAndContext(
    newTyp: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner,
    that: OrderedRVD,
    preservesPartitioning: Boolean = false
  )(zipper: (RVDContext, RVDContext => Iterator[RegionValue], RVDContext => Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = OrderedRVD(
    newTyp,
    newPartitioner,
    crdd.czipPartitionsAndContext(that.crdd, preservesPartitioning) { (ctx, lit, rit) =>
      zipper(ctx, ctx => lit.flatMap(_ (ctx)), ctx => rit.flatMap(_ (ctx)))
    }
  )

  def zipPartitionsPreservesPartitioning[T: ClassTag](
    newTyp: OrderedRVDType,
    that: ContextRDD[RVDContext, T]
  )(zipper: (Iterator[RegionValue], Iterator[T]) => Iterator[RegionValue]
  ): OrderedRVD = OrderedRVD(
    newTyp,
    partitioner,
    crdd.zipPartitions(that)(zipper))

  def zipPartitions(
    newTyp: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner,
    that: RVD
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = zipPartitions(newTyp, newPartitioner, that, false)(zipper)

  def zipPartitions(
    newTyp: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner,
    that: RVD,
    preservesPartitioning: Boolean
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = OrderedRVD(
    newTyp,
    newPartitioner,
    boundary.crdd.czipPartitions(that.boundary.crdd, preservesPartitioning)(zipper))

  def zipPartitions[T: ClassTag](
    that: RVD
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[T]
  ): ContextRDD[RVDContext, T] = zipPartitions(that, false)(zipper)

  def zipPartitions[T: ClassTag](
    that: RVD,
    preservesPartitioning: Boolean
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[T]
  ): ContextRDD[RVDContext, T] =
    boundary.crdd.czipPartitions(that.boundary.crdd, preservesPartitioning)(zipper)

  def zip(
    newTyp: OrderedRVDType,
    that: RVD
  )(zipper: (RVDContext, RegionValue, RegionValue) => RegionValue
  ): OrderedRVD = OrderedRVD(
    newTyp,
    partitioner,
    this.crdd.czip(that.crdd, preservesPartitioning = true)(zipper))

  def writeRowsSplit(
    path: String,
    t: MatrixType,
    codecSpec: CodecSpec,
    stageLocally: Boolean
  ): Array[Long] = crdd.writeRowsSplit(path, t, codecSpec, partitioner, stageLocally)

  override def toUnpartitionedRVD: UnpartitionedRVD =
    new UnpartitionedRVD(typ.rowType, crdd)
}

object OrderedRVD {
  def empty(sc: SparkContext, typ: OrderedRVDType): OrderedRVD = {
    OrderedRVD(typ,
      OrderedRVDPartitioner.empty(typ),
      ContextRDD.empty[RVDContext, RegionValue](sc))
  }

  /**
    * Precondition: the iterator it is PK-sorted.  We lazily K-sort each block
    * of PK-equivalent elements.
    */
  def localKeySort(
    consumerRegion: Region,
    producerRegion: Region,
    ctx: RVDContext,
    typ: OrderedRVDType,
    // it: Iterator[RegionValue[rowType]]
    it: Iterator[RegionValue]
  ): Iterator[RegionValue] =
    new Iterator[RegionValue] {
      private val bit = it.buffered

      private val q = new mutable.PriorityQueue[RegionValue]()(typ.kInRowOrd.reverse)

      private val rvb = new RegionValueBuilder(consumerRegion)

      private val rv = RegionValue()

      def hasNext: Boolean = bit.hasNext || q.nonEmpty

      def next(): RegionValue = {
        if (q.isEmpty) {
          do {
            val rv = bit.next()
            val r = ctx.freshRegion
            rvb.set(r)
            rvb.start(typ.rowType)
            rvb.addRegionValue(typ.rowType, rv)
            q.enqueue(RegionValue(rvb.region, rvb.end()))
            producerRegion.clear()
          } while (bit.hasNext && typ.pkInRowOrd.compare(q.head, bit.head) == 0)
        }

        rvb.set(consumerRegion)
        rvb.start(typ.rowType)
        val fromQueue = q.dequeue()
        rvb.addRegionValue(typ.rowType, fromQueue)
        ctx.closeChild(fromQueue.region)
        rv.set(consumerRegion, rvb.end())
        rv
      }
    }

  def getKeys(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): ContextRDD[RVDContext, RegionValue] = {
    // The region values in 'crdd' are of type `typ.rowType`
    val localType = typ
    crdd.cmapPartitions { (ctx, it) =>
      val wrv = WritableRegionValue(localType.kType, ctx.freshRegion)
      it.map { rv =>
        wrv.setSelect(localType.rowType, localType.kRowFieldIdx, rv)
        wrv.value
      }
    }
  }

  def getPartitionKeyInfo(
    typ: OrderedRVDType,
    keys: ContextRDD[RVDContext, RegionValue]
  ): Array[OrderedRVPartitionInfo] = {
    // the region values in 'keys' are of typ `typ.keyType`
    val nPartitions = keys.getNumPartitions
    if (nPartitions == 0)
      return Array()

    val rng = new java.util.Random(1)
    val partitionSeed = Array.fill[Int](nPartitions)(rng.nextInt())

    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val localType = typ

    val pkis = keys.cmapPartitionsWithIndex { (i, ctx, it) =>
      val out = if (it.hasNext)
        Iterator(OrderedRVPartitionInfo(localType, samplesPerPartition, i, it, partitionSeed(i), ctx))
      else
        Iterator()
      out
    }.collect()

    pkis.sortBy(_.min)(typ.pkType.ordering.toOrdering)
  }

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD
  ): OrderedRVD = coerce(typ, rvd, None, None)

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD,
    fastKeys: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = coerce(typ, rvd, Some(fastKeys), None)

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD,
    hintPartitioner: OrderedRVDPartitioner
  ): OrderedRVD = coerce(typ, rvd, None, Some(hintPartitioner))

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD,
    fastKeys: Option[ContextRDD[RVDContext, RegionValue]],
    hintPartitioner: Option[OrderedRVDPartitioner]
  ): OrderedRVD = coerce(typ, rvd.crdd, fastKeys, hintPartitioner)

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
  ): OrderedRVD = coerce(
    typ,
    rdd,
    Some(fastKeys),
    Some(hintPartitioner))

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue],
    fastKeys: Option[RDD[RegionValue]],
    hintPartitioner: Option[OrderedRVDPartitioner]
  ): OrderedRVD = coerce(
    typ,
    ContextRDD.weaken[RVDContext](rdd),
    fastKeys.map(ContextRDD.weaken[RVDContext](_)),
    hintPartitioner)

  def coerce(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = coerce(typ, crdd, None, None)

  def coerce(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue],
    fastKeys: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = coerce(typ, crdd, Some(fastKeys), None)

  def makeCoercer(
    fullType: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue],
    hintPartitioner: Option[OrderedRVDPartitioner]): RVDCoercer = {
    val sc = crdd.sparkContext

    val emptyCoercer: RVDCoercer = new RVDCoercer(fullType) {
      def _coerce(typ: OrderedRVDType, crdd: ContextRDD[RVDContext, RegionValue]): OrderedRVD = empty(sc, typ)
    }

    if (crdd.partitions.isEmpty)
      return emptyCoercer

    // keys: RDD[RegionValue[kType]]
    val keys = crdd

    val pkis = getPartitionKeyInfo(fullType, keys)

    if (pkis.isEmpty)
      return emptyCoercer

    val partitionsSorted = (pkis, pkis.tail).zipped.forall { case (p, pnext) =>
      val r = fullType.pkType.ordering.lteq(p.max, pnext.min)
      if (!r)
        log.info(s"not sorted: p = $p, pnext = $pnext")
      r
    }

    val sortedness = pkis.map(_.sortedness).min
    if (partitionsSorted && sortedness >= OrderedRVPartitionInfo.TSORTED) {
      val (makeAdjustments, rangeBounds, adjSortedness) = rangesAndAdjustments(fullType, pkis, sortedness)

      val partitioner = new OrderedRVDPartitioner(
        fullType.partitionKey,
        fullType.kType,
        rangeBounds)

      (adjSortedness: @unchecked) match {
        case OrderedRVPartitionInfo.KSORTED =>
          info("Coerced sorted dataset")
          new RVDCoercer(fullType) {
            def _coerce(typ: OrderedRVDType, crdd: ContextRDD[RVDContext, RegionValue]): OrderedRVD = {
              OrderedRVD(
                typ,
                partitioner,
                crdd
                  .reorderPartitions(pkis.map(_.partitionIndex))
                  .adjustPartitions(makeAdjustments(typ)))
            }
          }

        case OrderedRVPartitionInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          new RVDCoercer(fullType) {
            def _coerce(typ: OrderedRVDType, crdd: ContextRDD[RVDContext, RegionValue]): OrderedRVD = {
              OrderedRVD(
                typ,
                partitioner,
                crdd
                  .reorderPartitions(pkis.map(_.partitionIndex))
                  .adjustPartitions(makeAdjustments(typ))
                  .cmapPartitionsAndContext { (consumerCtx, it) =>
                    val producerCtx = consumerCtx.freshContext
                    localKeySort(consumerCtx.region, producerCtx.region, consumerCtx, typ, it.flatMap(_ (producerCtx)))
                  })
            }
          }
      }
    } else {
      info("Ordering unsorted dataset with network shuffle")
      val partitioner = hintPartitioner
        .filter(_.numPartitions >= crdd.partitions.length)
        .getOrElse(new OrderedRVDPartitioner(
          fullType.partitionKey,
          fullType.kType,
          calculateKeyRanges(fullType, pkis, crdd.getNumPartitions)))

      new RVDCoercer(fullType) {
        def _coerce(typ: OrderedRVDType, crdd: ContextRDD[RVDContext, RegionValue]): OrderedRVD = {
          shuffle(typ, partitioner, crdd)
        }
      }
    }
  }

  def coerce(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue],
    fastKeys: Option[ContextRDD[RVDContext, RegionValue]],
    hintPartitioner: Option[OrderedRVDPartitioner]
  ): OrderedRVD = {
    val keys = fastKeys.getOrElse(getKeys(typ, crdd))
    val coercer = makeCoercer(typ, keys, hintPartitioner)
    coercer.coerce(typ, crdd)
  }

  def calculateKeyRanges(typ: OrderedRVDType, pkis: Array[OrderedRVPartitionInfo], nPartitions: Int): Array[Interval] = {
    assert(nPartitions > 0)
    assert(pkis.nonEmpty)

    val pkOrd = typ.pkType.ordering.toOrdering
    val keys = pkis
      .flatMap(_.samples)
      .sorted(pkOrd)

    val min = pkis.map(_.min).min(pkOrd)
    val max = pkis.map(_.max).max(pkOrd)

    val ab = new ArrayBuilder[Any]()
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

  def adjustBoundsAndShuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rvd: RVD
  ): OrderedRVD = {
    assert(typ.rowType == rvd.rowType)
    adjustBoundsAndShuffle(typ, partitioner, rvd.crdd)
  }

  private[this] def adjustBoundsAndShuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = {
    val pkType = partitioner.pkType
    val pkOrd = pkType.ordering.toOrdering
    val pkis = getPartitionKeyInfo(typ, getKeys(typ, crdd))

    if (pkis.isEmpty)
      return OrderedRVD(typ, partitioner, crdd)

    val min = pkis.map(_.min).min(pkOrd)
    val max = pkis.map(_.max).max(pkOrd)

    shuffle(typ, partitioner.enlargeToRange(Interval(min, max, true, true)), crdd)
  }

  def shuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rvd: RVD
  ): OrderedRVD = shuffle(typ, partitioner, rvd.crdd)

  def shuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = {
    val localType = typ
    val partBc = partitioner.broadcast(crdd.sparkContext)
    val enc = RVD.wireCodec.buildEncoder(localType.rowType)
    val dec = RVD.wireCodec.buildDecoder(localType.rowType, localType.rowType)
    OrderedRVD(typ,
      partitioner,
      crdd.cmapPartitions { (ctx, it) =>
        it.map { rv =>
          val keys: Any = SafeRow.selectFields(localType.rowType, rv)(localType.kRowFieldIdx)
          val bytes = RVD.regionValueToBytes(enc, ctx)(rv)
          (keys, bytes)
        }
      }.shuffle(partitioner.sparkPartitioner(crdd.sparkContext), typ.kType.ordering.toOrdering)
        .cmapPartitionsWithIndex { case (i, ctx, it) =>
          val region = ctx.region
          val rv = RegionValue(region)
          it.map { case (k, bytes) =>
            assert(partBc.value.getSafePartition(k) == i)
            RVD.bytesToRegionValue(dec, region, rv)(bytes)
          }
        })
  }

  def rangesAndAdjustments(fullType: OrderedRVDType,
    sortedKeyInfo: Array[OrderedRVPartitionInfo],
    sortedness: Int): (OrderedRVDType => Array[Array[Adjustment[RegionValue]]], Array[Interval], Int) = {

    val partitionBounds = new ArrayBuilder[Any]()
    val adjustmentsBuffer = new mutable.ArrayBuffer[Array[(Int, OrderedRVDType => Adjustment[RegionValue])]]
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
      val pkOrd = fullType.pkType.ordering

      indicesBuilder += i

      var continue = true
      while (continue && it.hasNext && pkOrd.equiv(sortedKeyInfo(it.head).min, max)) {
        anyOverlaps = true
        if (pkOrd.equiv(sortedKeyInfo(it.head).max, max))
          indicesBuilder += it.next()
        else {
          indicesBuilder += it.head
          continue = false
        }
      }

      val bufferSize = adjustmentsBuffer.size
      val adjustments = indicesBuilder.result().zipWithIndex.map { case (partitionIndex, index) =>
        assert(sortedKeyInfo(partitionIndex).sortedness >= OrderedRVPartitionInfo.TSORTED)
        (partitionIndex, (typ: OrderedRVDType) => {
          val f: Iterator[RegionValue] => Iterator[RegionValue] =
          // In the first adjusted partition, drop elements that belong in the previous adjusted partition
            if (index == 0)
              if (bufferSize > 0 && pkOrd.equiv(min, sortedKeyInfo(adjustmentsBuffer(bufferSize - 1).head._1).max)) {
                it: Iterator[RegionValue] =>
                  val itRegion = Region()
                  val rvb = new RegionValueBuilder(itRegion)
                  rvb.start(typ.pkType)
                  rvb.addAnnotation(typ.pkType, min)
                  val rvMin = RegionValue(itRegion, rvb.end())
                  it.dropWhile { rv =>
                    typ.pkRowOrd.compare(rvMin, rv) == 0
                  }
              } else
                identity
            else
            // In every subsequent partition, only take elements that are the max of the last
            { it: Iterator[RegionValue] =>
              val itRegion = Region()
              val rvb = new RegionValueBuilder(itRegion)
              rvb.start(typ.pkType)
              rvb.addAnnotation(typ.pkType, max)
              val rvMax = RegionValue(itRegion, rvb.end())
              it.takeWhile { rv =>
                typ.pkRowOrd.compare(rvMax, rv) == 0
              }
            }
          Adjustment(partitionIndex, f)
        })
      }

      adjustmentsBuffer += adjustments
      partitionBounds += max
    }

    val pBounds = partitionBounds.result()

    val rangeBounds = OrderedRVDPartitioner.makeRangeBoundIntervals(fullType.pkType, pBounds)

    val adjSortedness = if (anyOverlaps)
      sortedness.min(OrderedRVPartitionInfo.TSORTED)
    else
      sortedness

    ((typ: OrderedRVDType) => adjustmentsBuffer.toArray.map(_.map(_._2.apply(typ))), rangeBounds, adjSortedness)
  }

  def apply(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rvd: RVD
  ): OrderedRVD = apply(typ, partitioner, rvd.crdd)

  def apply(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    codec: CodecSpec,
    rdd: RDD[Array[Byte]]
  ): OrderedRVD = {
    val dec = codec.buildDecoder(typ.rowType, typ.rowType)
    apply(
      typ,
      partitioner,
      ContextRDD.weaken[RVDContext](rdd).cmapPartitions { (ctx, it) =>
        val rv = RegionValue()
        it.map(RVD.bytesToRegionValue(dec, ctx.region, rv))
      })
  }

  def apply(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = {
    val sc = crdd.sparkContext

    val partitionerBc = partitioner.broadcast(sc)
    val localType = typ

    new OrderedRVD(
      typ,
      partitioner,
      crdd.cmapPartitionsWithIndex { case (i, ctx, it) =>
        val prevK = WritableRegionValue(typ.kType, ctx.freshRegion)
        val prevPK = WritableRegionValue(typ.pkType, ctx.freshRegion)
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
            if (!partitionerBc.value.rangeBounds(i).contains(localType.pkType.ordering, pkUR)) {
              val shouldBeIn = partitionerBc.value.getPartitionPK(pkUR)
              fatal(
                s"""OrderedRVD error! Unexpected PK in partition $i
                   |  Range bounds for partition $i: ${ partitionerBc.value.rangeBounds(i) }
                   |  Key should be in partition ${ shouldBeIn }: ${ partitionerBc.value.rangeBounds(shouldBeIn) }
                   |  Invalid PK: ${ pkUR.toString() }
                   |  Full key: ${ new UnsafeRow(typ.kType, rv).toString() }""".stripMargin)
            }

            assert(localType.pkRowOrd.compare(prevPK.value, rv) == 0)
            rv
          }
        }
      })
  }

  def union(rvds: Seq[OrderedRVD]): OrderedRVD = {
    require(rvds.length > 1)
    val first = rvds.head
    OrderedRVD.coerce(
      first.typ,
      RVD.union(rvds),
      None,
      None)
  }
}
