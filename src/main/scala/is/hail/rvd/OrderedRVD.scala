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

  require(typ.kType isIsomorphicTo partitioner.kType)

  def boundary: OrderedRVD = OrderedRVD(typ, partitioner, crddBoundary)

  def rowType: TStruct = typ.rowType

  def updateType(newTyp: OrderedRVDType): OrderedRVD =
    OrderedRVD(newTyp, partitioner, crdd)

  def mapPreservesPartitioning(
    newTyp: OrderedRVDType
  )(f: (RegionValue) => RegionValue
  ): OrderedRVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    OrderedRVD(newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.map(f))
  }

  def mapPartitionsWithIndexPreservesPartitioning(
    newTyp: OrderedRVDType
  )(f: (Int, Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    OrderedRVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.mapPartitionsWithIndex(f))
  }

  def mapPartitionsWithIndexPreservesPartitioning(
    newTyp: OrderedRVDType,
    f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    OrderedRVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithIndex(f))
  }

  def mapPartitionsPreservesPartitioning(
    newTyp: OrderedRVDType
  )(f: (Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    OrderedRVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.mapPartitions(f))
  }

  def mapPartitionsPreservesPartitioning(
    newTyp: OrderedRVDType,
    f: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    OrderedRVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitions(f))
  }

  def extendKeyPreservesPartitioning(newKey: Array[String]): OrderedRVD = {
    require(newKey startsWith typ.key)
    require(newKey.forall(typ.rowType.fieldNames.contains))
    val orvdType = typ.copy(key = newKey)
    if (OrderedRVDPartitioner.isValid(orvdType.kType, partitioner.rangeBounds))
      copy(typ = orvdType)
    else
      constrainToOrderedPartitioner(orvdType, partitioner.extendKey(orvdType.kType))
  }

  def truncateKey(newKey: Array[String]): OrderedRVD = {
    require(typ.key startsWith newKey)
    copy(
      typ = typ.copy(key = newKey),
      partitioner = partitioner.coarsen(newKey.length))
  }

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

  def zipWithIndex(name: String, partitionCounts: Option[IndexedSeq[Long]] = None): OrderedRVD = {
    assert(!typ.key.contains(name))
    val (newRowType, newCRDD) = zipWithIndexCRDD(name, partitionCounts)

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

  // FIXME: don't allow to change type
  def constrainToOrderedPartitioner(
    ordType: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner
  ): OrderedRVD = {
    // FIXME: require newPartitioner disjoint
    require(newPartitioner.kType isPrefixOf typ.kType)

    new OrderedRVD(
      typ.copy(key = typ.key.take(newPartitioner.kType.size)),
      newPartitioner,
      ContextRDD(new RepartitionedOrderedRDD2(this, newPartitioner.rangeBounds)))
  }

  def localSort(newKey: Array[String]): OrderedRVD = {
    require(newKey startsWith typ.key)
    require(newKey.forall(typ.rowType.fieldNames.contains))
    require(partitioner.satisfiesPartitionKey(typ.key.length - 1))

    val (newKType, _) = typ.rowType.select(newKey)
    val oldType = typ
    val newType = typ.copy(key = newKey)
    val newPartitioner = partitioner.copy(kType = newKType)
    val sortedRDD = crdd.cmapPartitionsAndContext { (consumerCtx, it) =>
      val producerCtx = consumerCtx.freshContext
      OrderedRVD.localKeySort(
        consumerCtx.region,
        producerCtx.region,
        consumerCtx,
        oldType,
        newType,
        it.flatMap(_ (producerCtx)))
    }

    new OrderedRVD(newType, newPartitioner, sortedRDD)
  }

  def keyBy(key: Array[String] = typ.key): KeyedOrderedRVD =
    new KeyedOrderedRVD(this, key)

  def orderedLeftJoinDistinctAndInsert(
    right: OrderedRVD,
    root: String,
    lift: Option[String] = None,
    dropLeft: Option[Array[String]] = None
  ): OrderedRVD = {
    assert(!typ.key.contains(root))

    val valueStruct = right.typ.valueType
    val rightRowType = right.typ.rowType
    val liftField = lift.map { name => rightRowType.field(name) }
    val valueType = liftField.map(-_.typ).getOrElse(valueStruct)

    val removeLeft = dropLeft.map(_.toSet).getOrElse(Set.empty)
    val keepIndices = rowType.fields.filter(f => !removeLeft.contains(f.name)).map(_.index).toArray
    val newRowType = TStruct(
      keepIndices.map(i => rowType.fieldNames(i) -> rowType.types(i)) ++ Array(root -> valueType): _*)

    val localRowType = rowType

    val shouldLift = lift.isDefined
    val rightValueIndices = right.typ.valueIndices
    val liftIndex = liftField.map(_.index).getOrElse(-1)

    val joiner = { (ctx: RVDContext, it: Iterator[JoinedRegionValue]) =>
      val rvb = ctx.rvb
      val rv = RegionValue()

      it.map { jrv =>
        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight
        rvb.start(newRowType)
        rvb.startStruct()
        rvb.addFields(localRowType, lrv, keepIndices)
        if (rrv == null)
          rvb.setMissing()
        else {
          if (shouldLift)
            rvb.addField(rightRowType, rrv, liftIndex)
          else {
            rvb.startStruct()
            rvb.addFields(rightRowType, rrv, rightValueIndices)
            rvb.endStruct()
          }
        }
        rvb.endStruct()
        rv.set(ctx.region, rvb.end())
        rv
      }
    }
    assert(typ.key.length >= right.typ.key.length, s"$typ >= ${ right.typ }\n  $this\n  $right")
    keyBy(typ.key.take(right.typ.key.length)).orderedJoinDistinct(
      right.keyBy(),
      "left",
      joiner,
      typ.copy(rowType = newRowType,
        key = typ.key.filter(!removeLeft.contains(_)))
    )
  }

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

  def orderedMerge(right: OrderedRVD): OrderedRVD =
    keyBy().orderedMerge(right.keyBy())

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

  def copy(
    typ: OrderedRVDType = typ,
    partitioner: OrderedRVDPartitioner = partitioner,
    crdd: ContextRDD[RVDContext, RegionValue] = crdd
  ): OrderedRVD =
    OrderedRVD(typ, partitioner, crdd)

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
      val pki = OrderedRVD.getKeyInfo(typ, typ.key.length, OrderedRVD.getKeys(typ, shuffled))
      if (pki.isEmpty)
        return OrderedRVD.empty(sparkContext, typ)
      val partitioner = OrderedRVD.calculateKeyRanges(
        typ, pki, shuffled.getNumPartitions, typ.key.length)
      OrderedRVD.shuffle(typ, partitioner, shuffled)
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
    val kType = typ.kType
    val kRowFieldIdx = typ.kRowFieldIdx
    val rowType = typ.rowType

    mapPartitionsPreservesPartitioning(typ, { (ctx, it) =>
      val kUR = new UnsafeRow(kType)
      it.filter { rv =>
        ctx.rvb.start(kType)
        ctx.rvb.selectRegionValue(rowType, kRowFieldIdx, rv)
        kUR.set(ctx.region, ctx.rvb.end())
        !intervalsBc.value.contains(kType.ordering, kUR)
      }
    })
  }

  def filterToIntervals(intervals: IntervalTree[_]): OrderedRVD = {
    val kOrdering = typ.kType.ordering
    val intervalsBc = crdd.sparkContext.broadcast(intervals)
    val rowType = typ.rowType
    val kRowFieldIdx = typ.kRowFieldIdx

    val pred: (RegionValue) => Boolean = (rv: RegionValue) => {
      val ur = new UnsafeRow(rowType, rv)
      val key = Row.fromSeq(
        kRowFieldIdx.map(i => ur.get(i)))
      intervalsBc.value.contains(kOrdering, key)
    }

    val nPartitions = getNumPartitions
    if (nPartitions <= 1)
      return filter(pred)

    val newPartitionIndices = intervals.toIterator.flatMap { case (i, _) =>
      partitioner.getPartitionRange(i)
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

  def head(n: Long, partitionCounts: Option[IndexedSeq[Long]]): OrderedRVD = {
    require(n >= 0)

    if (n == 0)
      return OrderedRVD.empty(sparkContext, typ)

    val newRDD = crdd.head(n, partitionCounts)
    val newNParts = newRDD.getNumPartitions
    assert(newNParts >= 0)

    val newRangeBounds = Array.range(0, newNParts).map(partitioner.rangeBounds)
    val newPartitioner = partitioner.copy(rangeBounds = newRangeBounds)

    OrderedRVD(typ, newPartitioner, newRDD)
  }

  def groupByKey(valuesField: String = "values"): OrderedRVD = {
    val newTyp = new OrderedRVDType(
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

    val newPartitioner = partitioner.copy(rangeBounds = keep.map(partitioner.rangeBounds))

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

  // New key type must be prefix of left key type. 'joinKey' must be prefix of
  // both left key and right key. 'zipper' must take all output key values from
  // left iterator, and be monotonic on left iterator (it can drop or duplicate
  // elements of left iterator, or insert new elements in order, but cannot
  // rearrange them), and output region values must conform to 'newTyp'. The
  // partitioner of the resulting OrderedRVD will be left partitioner truncated
  // to new key. Each partition will be computed by 'zipper', with corresponding
  // partition of 'this' as first iterator, and with all rows of 'that' whose
  // 'joinKey' might match something in partition as the second iterator.
  def alignAndZipPartitions(
    newTyp: OrderedRVDType,
    that: OrderedRVD,
    joinKey: TStruct
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): OrderedRVD = {
    require(newTyp.kType isPrefixOf this.typ.kType)
    require(joinKey isPrefixOf this.typ.kType)
    require(joinKey isPrefixOf that.typ.kType)

    val left = this.truncateKey(newTyp.key)
    val coarsenedPartitioner = this.partitioner.coarsen(joinKey.size)
    OrderedRVD(
      typ = newTyp,
      partitioner = left.partitioner,
      crdd = left.crddBoundary.czipPartitions(
        that.constrainToOrderedPartitioner(
          that.typ.copy(
            key = that.typ.key.take(coarsenedPartitioner.kType.size)),
          coarsenedPartitioner
        ).crddBoundary
      )(zipper))
  }

  def writeRowsSplit(
    path: String,
    t: MatrixType,
    codecSpec: CodecSpec,
    stageLocally: Boolean
  ): Array[Long] = crdd.writeRowsSplit(path, t, codecSpec, partitioner, stageLocally)

  override def toUnpartitionedRVD: RVD =
    UnpartitionedRVD(typ.rowType, crdd)

  def toOldStyleRVD: RVD =
    if (typ.key.isEmpty)
      toUnpartitionedRVD
    else
      this

  override def toOrderedRVD: OrderedRVD = this
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
    oldType: OrderedRVDType,
    newType: OrderedRVDType,
    // it: Iterator[RegionValue[rowType]]
    it: Iterator[RegionValue]
  ): Iterator[RegionValue] =
    new Iterator[RegionValue] {
      private val bit = it.buffered

      private val q = new mutable.PriorityQueue[RegionValue]()(newType.kInRowOrd.reverse)

      private val rvb = new RegionValueBuilder(consumerRegion)

      private val rv = RegionValue()

      def hasNext: Boolean = bit.hasNext || q.nonEmpty

      def next(): RegionValue = {
        if (q.isEmpty) {
          do {
            val rv = bit.next()
            val r = ctx.freshRegion
            rvb.set(r)
            rvb.start(newType.rowType)
            rvb.addRegionValue(newType.rowType, rv)
            q.enqueue(RegionValue(rvb.region, rvb.end()))
            producerRegion.clear()
          } while (bit.hasNext && oldType.kInRowOrd.compare(q.head, bit.head) == 0)
        }

        rvb.set(consumerRegion)
        rvb.start(newType.rowType)
        val fromQueue = q.dequeue()
        rvb.addRegionValue(newType.rowType, fromQueue)
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

  def getKeyInfo(
    typ: OrderedRVDType,
    partitionKey: Int,
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
        Iterator(OrderedRVPartitionInfo(localType, partitionKey, samplesPerPartition, i, it, partitionSeed(i), ctx))
      else
        Iterator()
      out
    }.collect()

    pkis.sortBy(_.min)(typ.kType.ordering.toOrdering)
  }

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD
  ): OrderedRVD = coerce(typ, rvd, None)

  def coerce(
    typ: OrderedRVDType,
    partitionKey: Int,
    rvd: RVD
  ): OrderedRVD = coerce(typ, partitionKey, rvd, None)

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD,
    fastKeys: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = coerce(typ, rvd, Some(fastKeys))

  def coerce(
    typ: OrderedRVDType,
    rvd: RVD,
    fastKeys: Option[ContextRDD[RVDContext, RegionValue]]
  ): OrderedRVD = coerce(typ, rvd.crdd, fastKeys)

  def coerce(
    typ: OrderedRVDType,
    partitionKey: Int,
    rvd: RVD,
    fastKeys: Option[ContextRDD[RVDContext, RegionValue]]
  ): OrderedRVD = coerce(typ, partitionKey, rvd.crdd, fastKeys)

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue]
  ): OrderedRVD = coerce(typ, rdd, None)

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue],
    fastKeys: RDD[RegionValue]
  ): OrderedRVD = coerce(typ, rdd, Some(fastKeys))

  def coerce(
    typ: OrderedRVDType,
    rdd: RDD[RegionValue],
    fastKeys: Option[RDD[RegionValue]]
  ): OrderedRVD = coerce(
    typ,
    ContextRDD.weaken[RVDContext](rdd),
    fastKeys.map(ContextRDD.weaken[RVDContext](_))
    )

  def coerce(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = coerce(typ, crdd, None)

  def coerce(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue],
    fastKeys: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = coerce(typ, crdd, Some(fastKeys))

  def coerce(
    typ: OrderedRVDType,
    crdd: ContextRDD[RVDContext, RegionValue],
    fastKeys: Option[ContextRDD[RVDContext, RegionValue]]
  ): OrderedRVD = {
    val keys = fastKeys.getOrElse(getKeys(typ, crdd))
    makeCoercer(typ, keys).coerce(typ, crdd)
  }

  def coerce(
    typ: OrderedRVDType,
    partitionKey: Int,
    crdd: ContextRDD[RVDContext, RegionValue],
    fastKeys: Option[ContextRDD[RVDContext, RegionValue]]
  ): OrderedRVD = {
    val keys = fastKeys.getOrElse(getKeys(typ, crdd))
    makeCoercer(typ, partitionKey, keys).coerce(typ, crdd)
  }

  def makeCoercer(
    fullType: OrderedRVDType,
    // keys: RDD[RegionValue[fullType.kType]]
    keys: ContextRDD[RVDContext, RegionValue]
  ): RVDCoercer = makeCoercer(fullType, fullType.key.length, keys)

  def makeCoercer(
    fullType: OrderedRVDType,
    partitionKey: Int,
    // keys: RDD[RegionValue[fullType.kType]]
    keys: ContextRDD[RVDContext, RegionValue]
   ): RVDCoercer = {
    type CRDD = ContextRDD[RVDContext, RegionValue]
    val sc = keys.sparkContext

    val emptyCoercer: RVDCoercer = new RVDCoercer(fullType) {
      def _coerce(typ: OrderedRVDType, crdd: CRDD): OrderedRVD = empty(sc, typ)
    }

    val pkis = getKeyInfo(fullType, partitionKey, keys)

    if (pkis.isEmpty)
      return emptyCoercer

    val bounds = pkis.map(_.interval).toFastIndexedSeq
    val pkBounds = bounds.map(_.coarsen(partitionKey))
    def orderPartitions = { crdd: CRDD =>
      val pids = pkis.map(_.partitionIndex)
      if (pids.isSorted && crdd.getNumPartitions == pids.length)
        crdd
      else
        crdd.reorderPartitions(pids)
    }
    val intraPartitionSortedness = pkis.map(_.sortedness).min

    if (intraPartitionSortedness == OrderedRVPartitionInfo.KSORTED
        && OrderedRVDPartitioner.isValid(fullType.kType, bounds)) {

      info("Coerced sorted dataset")

      new RVDCoercer(fullType) {
        def _coerce(typ: OrderedRVDType, crdd: CRDD): OrderedRVD = {
          val unfixedPartitioner = new OrderedRVDPartitioner(
            None, typ.kType, bounds)
          val newPartitioner = OrderedRVDPartitioner.generate(
            fullType.key.take(partitionKey), fullType.kType, bounds)
          val unfixedRVD = OrderedRVD(
            typ,
            unfixedPartitioner,
            orderPartitions(crdd))

          unfixedRVD.constrainToOrderedPartitioner(typ, newPartitioner)
        }
      }

    } else if (intraPartitionSortedness >= OrderedRVPartitionInfo.TSORTED
        && OrderedRVDPartitioner.isValid(fullType.kType.truncate(partitionKey), pkBounds)) {

      info("Coerced almost-sorted dataset")
      val unfixedPartitioner = new OrderedRVDPartitioner(
        fullType.kType.truncate(partitionKey),
        pkBounds
      )
      val newPartitioner = OrderedRVDPartitioner.generate(
        fullType.key.take(partitionKey),
        fullType.kType.truncate(partitionKey),
        pkBounds
      )

      new RVDCoercer(fullType) {
        def _coerce(typ: OrderedRVDType, crdd: CRDD): OrderedRVD = {
          val unfixedRVD = OrderedRVD(
            typ.copy(key = typ.key.take(partitionKey)),
            unfixedPartitioner,
            orderPartitions(crdd))

          unfixedRVD
            .constrainToOrderedPartitioner(typ.copy(key = typ.key.take(partitionKey)), newPartitioner)
            .localSort(typ.key)
        }
      }

    } else {

      info("Ordering unsorted dataset with network shuffle")
      val partitioner = calculateKeyRanges(fullType, pkis, keys.getNumPartitions, partitionKey)

      new RVDCoercer(fullType) {
        def _coerce(typ: OrderedRVDType, crdd: CRDD): OrderedRVD =
          shuffle(typ, partitioner, crdd)
      }
    }
  }

  def calculateKeyRanges(
    typ: OrderedRVDType,
    pInfo: Array[OrderedRVPartitionInfo],
    nPartitions: Int,
    partitionKey: Int
  ): OrderedRVDPartitioner = {
    assert(nPartitions > 0)
    assert(pInfo.nonEmpty)

    val kord = typ.kType.ordering.toOrdering
    val min = pInfo.map(_.min).min(kord)
    val max = pInfo.map(_.max).max(kord)
    val samples = pInfo.flatMap(_.samples)

    OrderedRVDPartitioner.fromSampleKeys(typ, min, max, samples, nPartitions, partitionKey)
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
    val kType = partitioner.kType
    val kOrd = kType.ordering.toOrdering
    val pkis = getKeyInfo(typ, typ.key.length, getKeys(typ, crdd))

    if (pkis.isEmpty)
      return OrderedRVD(typ, partitioner, crdd)

    val min = pkis.map(_.min).min(kOrd)
    val max = pkis.map(_.max).max(kOrd)

    shuffle(typ, partitioner.enlargeToRange(Interval(min, max, true, true)), crdd)
  }

  def shuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    rvd: RVD
  ): OrderedRVD = {
    require(typ.rowType == rvd.rowType)
    require(typ.kType isIsomorphicTo partitioner.kType)

    shuffle(typ, partitioner, rvd.crdd)
  }

  // Shuffles the data in 'crdd', producing an OrderedRVD with 'partitioner'.
  // WARNING: will drop any data with keys falling outside 'partitioner'.
  def shuffle(
    typ: OrderedRVDType,
    partitioner: OrderedRVDPartitioner,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): OrderedRVD = {
    val localType = typ
    val kOrdering = typ.kType.ordering

    val partBc = partitioner.broadcast(crdd.sparkContext)
    val enc = RVD.wireCodec.buildEncoder(localType.rowType)
    val dec = RVD.wireCodec.buildDecoder(localType.rowType, localType.rowType)

    val shuffledCRDD = crdd
      .mapPartitions { it =>
        val ur = new UnsafeRow(typ.rowType, null, 0)
        val key = new KeyedRow(ur, typ.kRowFieldIdx)
        it.filter { rv =>
          ur.set(rv)
          partBc.value.rangeTree.contains(kOrdering, key)
        }
      }
      .cmapPartitions { (ctx, it) =>
        it.map { rv =>
          val keys: Any = SafeRow.selectFields(localType.rowType, rv)(localType.kRowFieldIdx)
          val bytes = RVD.regionValueToBytes(enc, ctx)(rv)
          (keys, bytes)
        }
      }
      .shuffle(partitioner.sparkPartitioner(crdd.sparkContext), typ.kType.ordering.toOrdering)
      .cmapPartitionsWithIndex { case (i, ctx, it) =>
        val region = ctx.region
        val rv = RegionValue(region)
        it.map { case (k, bytes) =>
          assert(partBc.value.contains(i, k))
          RVD.bytesToRegionValue(dec, region, rv)(bytes)
        }
      }

    OrderedRVD(typ, partitioner, shuffledCRDD)
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
        val kUR = new UnsafeRow(typ.kType)

        new Iterator[RegionValue] {
          var first = true

          def hasNext: Boolean = it.hasNext

          def next(): RegionValue = {
            val rv = it.next()

            if (first)
              first = false
            else {
              assert(localType.kRowOrd.lteq(prevK.value, rv))
            }

            prevK.setSelect(localType.rowType, localType.kRowFieldIdx, rv)
            kUR.set(prevK.value)

            if (!partitionerBc.value.rangeBounds(i).contains(localType.kType.ordering, kUR))
              fatal(
                s"""OrderedRVD error! Unexpected key in partition $i
                   |  Range bounds for partition $i: ${ partitionerBc.value.rangeBounds(i) }
                   |  Invalid key: ${ kUR.toString() }""".stripMargin)

            rv
          }
        }
      })
  }

  def union(rvds: Seq[OrderedRVD]): OrderedRVD = {
    require(rvds.length > 1)
    rvds.reduce(_.orderedMerge(_))
  }
}
