package is.hail.rvd

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s.{theHailClassLoaderForSparkWorkers, HailClassLoader}
import is.hail.backend.{ExecuteContext, HailStateManager, HailTaskContext}
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.expr.ir.InferPType
import is.hail.expr.ir.PruneDeadFields.isSupertype
import is.hail.io._
import is.hail.io.index.IndexWriter
import is.hail.sparkextras._
import is.hail.types._
import is.hail.types.physical.{PCanonicalStruct, PInt64, PStruct}
import is.hail.types.virtual.{TInterval, TStruct}
import is.hail.utils._
import is.hail.utils.PartitionCounts.{getPCSubsetOffset, incrementalPCSubsetOffset, PCSubsetOffset}

import scala.reflect.ClassTag

import java.util

import org.apache.spark.{Partitioner, SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel

abstract class RVDCoercer(val fullType: RVDType) {
  final def coerce(typ: RVDType, crdd: ContextRDD[Long]): RVD = {
    require(isSupertype(typ.rowType.virtualType, fullType.rowType.virtualType))
    require(typ.key.sameElements(fullType.key))
    _coerce(typ, crdd)
  }

  protected def _coerce(typ: RVDType, crdd: ContextRDD[Long]): RVD
}

class RVD(
  val typ: RVDType,
  val partitioner: RVDPartitioner,
  val crdd: ContextRDD[Long],
) {
  self =>
  require(crdd.getNumPartitions == partitioner.numPartitions)

  require(typ.kType.virtualType isIsomorphicTo partitioner.kType)

  // Basic accessors

  def sparkContext: SparkContext = crdd.sparkContext

  def getNumPartitions: Int = crdd.getNumPartitions

  def rowType: TStruct = typ.rowType.virtualType

  def rowPType: PStruct = typ.rowType

  def boundary: RVD = RVD(typ, partitioner, crddBoundary)

  // Basic manipulators

  def cast(newRowType: PStruct): RVD = {
    val nameMap = rowType.fieldNames.zip(newRowType.fieldNames).toMap
    val newTyp = RVDType(newRowType, typ.key.map(nameMap))
    val newPartitioner = partitioner.rename(nameMap)
    new RVD(newTyp, newPartitioner, crdd)
  }

  // Exporting

  def toRows: RDD[Row] = {
    val localRowType = rowPType
    map((_, ptr) => SafeRow(localRowType, ptr))
  }

  def toUnsafeRows: RDD[UnsafeRow] = {
    val localRowPType = rowPType
    map((ctx, ptr) => new UnsafeRow(localRowPType, ctx.region, ptr))
  }

  def stabilize(ctx: ExecuteContext, enc: AbstractTypedCodecSpec): RDD[Array[Byte]] = {
    val makeEnc = enc.buildEncoder(ctx, rowPType)
    crdd.mapPartitions(it =>
      RegionValue.toBytes(theHailClassLoaderForSparkWorkers, makeEnc, it)
    ).run
  }

  def encodedRDD(ctx: ExecuteContext, enc: AbstractTypedCodecSpec): RDD[Array[Byte]] =
    stabilize(ctx, enc)

  def keyedEncodedRDD(
    ctx: ExecuteContext,
    enc: AbstractTypedCodecSpec,
    key: IndexedSeq[String] = typ.key,
  ): RDD[(Any, Array[Byte])] = {
    val makeEnc = enc.buildEncoder(ctx, rowPType)
    val kFieldIdx = typ.copy(key = key).kFieldIdx

    val localRowPType = rowPType
    crdd.cmapPartitions { (ctx, it) =>
      val encoder = new ByteArrayEncoder(theHailClassLoaderForSparkWorkers, makeEnc)
      TaskContext.get.addTaskCompletionListener[Unit](_ => encoder.close())
      it.map { ptr =>
        val keys: Any = SafeRow.selectFields(localRowPType, ctx.r, ptr)(kFieldIdx)
        val bytes = encoder.regionValueToBytes(ptr)
        (keys, bytes)
      }
    }.run
  }

  // Return an OrderedRVD whose key equals or at least starts with 'newKey'.
  def enforceKey(
    execCtx: ExecuteContext,
    newKey: IndexedSeq[String],
    isSorted: Boolean = false,
  ): RVD = {
    require(newKey.forall(rowType.hasField))
    val sharedPrefixLength = typ.key.zip(newKey).takeWhile { case (l, r) => l == r }.length
    if (isSorted && sharedPrefixLength == 0 && newKey.nonEmpty) {
      throw new IllegalArgumentException(
        s"$isSorted, $sharedPrefixLength, $newKey, $typ, $partitioner"
      )
    }

    if (sharedPrefixLength == newKey.length)
      this
    else if (isSorted)
      truncateKey(newKey.take(sharedPrefixLength))
        .extendKeyPreservesPartitioning(execCtx, newKey)
        .checkKeyOrdering()
    else
      changeKey(execCtx, newKey)
  }

  // Key and partitioner manipulation
  def changeKey(
    execCtx: ExecuteContext,
    newKey: IndexedSeq[String],
  ): RVD =
    changeKey(execCtx, newKey, newKey.length)

  def changeKey(
    execCtx: ExecuteContext,
    newKey: IndexedSeq[String],
    partitionKey: Int,
  ): RVD =
    RVD.coerce(execCtx, typ.copy(key = newKey), partitionKey, this.crdd)

  def extendKeyPreservesPartitioning(
    ctx: ExecuteContext,
    newKey: IndexedSeq[String],
  ): RVD = {
    require(newKey startsWith typ.key)
    require(newKey.forall(typ.rowType.fieldNames.contains))
    val rvdType = typ.copy(key = newKey)
    if (
      RVDPartitioner.isValid(ctx.stateManager, rvdType.kType.virtualType, partitioner.rangeBounds)
    )
      copy(typ = rvdType, partitioner = partitioner.copy(kType = rvdType.kType.virtualType))
    else {
      val adjustedPartitioner = partitioner.strictify()
      repartition(ctx, adjustedPartitioner)
        .copy(
          typ = rvdType,
          partitioner = adjustedPartitioner.copy(kType = rvdType.kType.virtualType),
        )
    }
  }

  def checkKeyOrdering(): RVD = {
    val partitionerBc = partitioner.broadcast(crdd.sparkContext)
    val localType = typ
    val localKPType = typ.kType
    val stateManager = partitioner.sm

    val ord = PartitionBoundOrdering(stateManager, localType.kType.virtualType)
    new RVD(
      typ,
      partitioner,
      crdd.cmapPartitionsWithIndex { case (i, ctx, it) =>
        val regionForWriting = ctx.freshRegion() // This one gets cleaned up when context is freed.
        val prevK = WritableRegionValue(stateManager, localType.kType, regionForWriting)
        val kUR = new UnsafeRow(localKPType)

        new Iterator[Long] {
          var first = true

          def hasNext: Boolean = it.hasNext

          def next(): Long = {
            val ptr = it.next()

            if (first)
              first = false
            else {
              if (localType.kRowOrd(stateManager).gt(prevK.value.offset, ptr)) {
                val prevKeyString = Region.pretty(localKPType, prevK.value.offset)

                prevK.setSelect(localType.rowType, localType.kFieldIdx, ptr, deepCopy = true)
                val currKeyString = Region.pretty(localKPType, prevK.value.offset)
                fatal(
                  s"""RVD error! Keys found out of order:
                     |  Current key:  $currKeyString
                     |  Previous key: $prevKeyString
                     |This error can occur after a split_multi if the dataset
                     |contains both multiallelic variants and duplicated loci.
                   """.stripMargin
                )
              }
            }

            prevK.setSelect(localType.rowType, localType.kFieldIdx, ptr, deepCopy = true)
            kUR.set(prevK.value)

            if (!partitionerBc.value.rangeBounds(i).contains(ord, kUR))
              fatal(
                s"""RVD error! Unexpected key in partition $i
                   |  Range bounds for partition $i: ${partitionerBc.value.rangeBounds(i)}
                   |  Range of partition IDs for key: [${partitionerBc.value.lowerBound(kUR)}, ${partitionerBc.value.upperBound(kUR)})
                   |  Invalid key: ${Region.pretty(localKPType, prevK.value.offset)}""".stripMargin
              )
            ptr
          }
        }
      },
    )
  }

  def truncateKey(n: Int): RVD = {
    require(n <= typ.key.length)
    truncateKey(typ.key.take(n))
  }

  def truncateKey(newKey: IndexedSeq[String]): RVD = {
    require(typ.key startsWith newKey)
    if (typ.key == newKey)
      this
    else
      copy(
        typ = typ.copy(key = newKey),
        partitioner = partitioner.coarsen(newKey.length),
      )
  }

  // WARNING: will drop any data with keys falling outside 'partitioner'.
  def repartition(
    ctx: ExecuteContext,
    newPartitioner: RVDPartitioner,
    shuffle: Boolean = false,
    filter: Boolean = true,
  ): RVD = {
    if (newPartitioner == this.partitioner)
      return this

    require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    require(shuffle || newPartitioner.kType.isPrefixOf(typ.kType.virtualType))

    if (shuffle) {
      val newType = typ.copy(key = newPartitioner.kType.fieldNames)

      val localRowPType = rowPType
      val kOrdering = PartitionBoundOrdering(ctx.stateManager, newType.kType.virtualType)

      val partBc = newPartitioner.broadcast(crdd.sparkContext)
      val enc = TypedCodecSpec(rowPType, BufferSpec.wireSpec)

      val filtered: RVD = if (filter) filterWithContext[(UnsafeRow, SelectFieldsRow)](
        { case (_, _) =>
          val ur = new UnsafeRow(localRowPType, null, 0)
          val key = new SelectFieldsRow(ur, newType.kFieldIdx)
          (ur, key)
        },
        { case ((ur, key), ctx, ptr) =>
          ur.set(ctx.r, ptr)
          partBc.value.contains(key)
        },
      )
      else this

      val shuffled: RDD[(Any, Array[Byte])] = new ShuffledRDD(
        filtered.keyedEncodedRDD(ctx, enc, newType.key),
        newPartitioner.sparkPartitioner(crdd.sparkContext),
      ).setKeyOrdering(kOrdering.toOrdering)

      val (rType: PStruct, shuffledCRDD) =
        enc.decodeRDD(ctx, localRowPType.virtualType, shuffled.values)

      RVD(RVDType(rType, newType.key), newPartitioner, shuffledCRDD)
    } else {
      if (newPartitioner != partitioner)
        new RVD(
          typ.copy(key = typ.key.take(newPartitioner.kType.size)),
          newPartitioner,
          RepartitionedOrderedRDD2(ctx.stateManager, this, newPartitioner.rangeBounds),
        )
      else
        this
    }
  }

  def naiveCoalesce(
    maxPartitions: Int,
    executeContext: ExecuteContext,
  ): RVD = {
    val n = partitioner.numPartitions
    if (maxPartitions >= n)
      return this

    val newN = maxPartitions
    val newNParts = partition(n, newN)
    assert(newNParts.forall(_ > 0))
    val newPartEnd = newNParts.scanLeft(-1)(_ + _).tail
    val newPartitioner = partitioner.coalesceRangeBounds(newPartEnd)

    if (newPartitioner == partitioner) {
      this
    } else {
      new RVD(
        typ,
        newPartitioner,
        crdd.coalesceWithEnds(newPartEnd),
      )
    }
  }

  def coalesce(
    ctx: ExecuteContext,
    maxPartitions: Int,
    shuffle: Boolean,
  ): RVD = {
    require(maxPartitions > 0, "cannot coalesce to nPartitions <= 0")
    val n = crdd.partitions.length
    if (!shuffle && maxPartitions >= n)
      return this

    if (shuffle) {
      val enc = TypedCodecSpec(rowPType, BufferSpec.wireSpec)
      val shuffledBytes = stabilize(ctx, enc).coalesce(maxPartitions, shuffle = true)
      val (newRowPType, shuffled) = destabilize(ctx, shuffledBytes, enc)
      if (typ.key.isEmpty)
        return RVD.unkeyed(newRowPType, shuffled)

      val newType = RVDType(newRowPType, typ.key)
      val keyInfo =
        RVD.getKeyInfo(ctx, newType, newType.key.length, RVD.getKeys(ctx, newType, shuffled))
      if (keyInfo.isEmpty)
        return RVD.empty(ctx, typ)
      val newPartitioner = RVD.calculateKeyRanges(
        ctx,
        newType,
        keyInfo,
        shuffled.getNumPartitions,
        newType.key.length,
      )

      if (newPartitioner.numPartitions < maxPartitions)
        warn(s"coalesced to ${newPartitioner.numPartitions} " +
          s"${plural(newPartitioner.numPartitions, "partition")}, less than requested $maxPartitions")

      repartition(ctx, newPartitioner, shuffle)
    } else {
      val partSize = countPerPartition()
      log.info(s"partSize = ${partSize.toSeq}")

      val partCumulativeSize =
        mapAccumulate[Array, Long](partSize, 0L)((s, acc) => (s + acc, s + acc))
      val totalSize = partCumulativeSize.last

      var newPartEnd = (0 until maxPartitions).map { i =>
        val t = totalSize * (i + 1) / maxPartitions

        /* j largest index not greater than t */
        var j = util.Arrays.binarySearch(partCumulativeSize, t)
        if (j < 0)
          j = -j - 1
        while (
          j < partCumulativeSize.length - 1
          && partCumulativeSize(j + 1) == t
        )
          j += 1
        assert(t <= partCumulativeSize(j) &&
          (j == partCumulativeSize.length - 1 ||
            t < partCumulativeSize(j + 1)))
        j
      }.toArray

      newPartEnd = newPartEnd.zipWithIndex.filter { case (_, i) =>
        i == 0 || newPartEnd(i) != newPartEnd(i - 1)
      }
        .map(_._1)

      val newPartitioner = partitioner.coalesceRangeBounds(newPartEnd)
      if (newPartitioner.numPartitions < maxPartitions)
        warn(s"coalesced to ${newPartitioner.numPartitions} " +
          s"${plural(newPartitioner.numPartitions, "partition")}, less than requested $maxPartitions")

      if (newPartitioner == partitioner) {
        this
      } else {
        new RVD(
          typ,
          newPartitioner,
          crdd.coalesceWithEnds(newPartEnd),
        )
      }
    }
  }

  // Key-wise operations

  def distinctByKey(execCtx: ExecuteContext): RVD = {
    val sm = execCtx.stateManager
    val localType = typ
    repartition(execCtx, partitioner.strictify())
      .mapPartitions(typ)((ctx, it) =>
        OrderedRVIterator(localType, it.toIteratorRV(ctx.r), ctx, sm)
          .staircase
          .map(_.value.offset)
      )
  }

  def localSort(newKey: IndexedSeq[String]): RVD = {
    require(newKey startsWith typ.key)
    require(newKey.forall(typ.rowType.fieldNames.contains))
    require(partitioner.satisfiesAllowedOverlap(typ.key.length - 1))

    val localTyp = typ
    val localSm = partitioner.sm
    val sortedRDD = crdd.toCRDDRegionValue.cmapPartitions { (consumerCtx, it) =>
      OrderedRVIterator(localTyp, it, consumerCtx, localSm).localKeySort(newKey)
    }.toCRDDPtr

    val newType = typ.copy(key = newKey)
    new RVD(
      newType,
      partitioner.copy(kType = newType.kType.virtualType),
      sortedRDD,
    )
  }

  // Mapping

  // None of the mapping methods may modify values of key fields, so that the
  // partitioner remains valid.

  def map[T](f: (RVDContext, Long) => T)(implicit tct: ClassTag[T]): RDD[T] =
    crdd.cmap(f).run

  def map(newTyp: RVDType)(f: (RVDContext, Long) => Long): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(newTyp, partitioner.coarsen(newTyp.key.length), crdd.cmap(f))
  }

  def mapPartitions[T: ClassTag](
    f: (RVDContext, Iterator[Long]) => Iterator[T]
  ): RDD[T] = crdd.cmapPartitions(f).run

  def mapPartitions(
    newTyp: RVDType
  )(
    f: (RVDContext, Iterator[Long]) => Iterator[Long]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitions(f),
    )
  }

  def mapPartitionsWithContext(
    newTyp: RVDType
  )(
    f: (RVDContext, RVDContext => Iterator[Long]) => Iterator[Long]
  ): RVD =
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithContext(f),
    )

  def mapPartitionsWithContextAndIndex(
    newTyp: RVDType
  )(
    f: (Int, RVDContext, RVDContext => Iterator[Long]) => Iterator[Long]
  ): RVD =
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithContextAndIndex(f),
    )

  def mapPartitionsWithIndex[T: ClassTag](
    f: (Int, RVDContext, Iterator[Long]) => Iterator[T]
  ): RDD[T] = crdd.cmapPartitionsWithIndex(f).run

  def mapPartitionsWithIndex(
    newTyp: RVDType
  )(
    f: (Int, RVDContext, Iterator[Long]) => Iterator[Long]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithIndex(f),
    )
  }

  def mapPartitionsWithIndexAndValue[V](
    newTyp: RVDType,
    values: Array[V],
  )(
    f: (Int, RVDContext, V, Iterator[Long]) => Iterator[Long]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithIndexAndValue(values, f),
    )
  }

  // Filtering

  def head(n: Long, partitionCounts: Option[IndexedSeq[Long]]): RVD = {
    require(n >= 0)

    if (n == 0)
      return RVD.empty(partitioner.sm, typ)

    val (idxLast, nTake) = partitionCounts match {
      case Some(pcs) =>
        getPCSubsetOffset(n, pcs.iterator) match {
          case Some(PCSubsetOffset(idx, nTake, _)) => idx -> nTake
          case None => return this
        }
      case None =>
        val crddCleanup = crdd.cleanupRegions
        val PCSubsetOffset(idx, nTake, _) =
          incrementalPCSubsetOffset(n, 0 until getNumPartitions)(
            crddCleanup.runJob(getIteratorSizeWithMaxN(n), _)
          )
        idx -> nTake
    }

    val newRDD = crdd.mapPartitionsWithIndex(
      { case (i, it) =>
        if (i == idxLast)
          it.take(nTake.toInt)
        else
          it
      },
      preservesPartitioning = true,
    )
      .subsetPartitions((0 to idxLast).toArray)

    val newNParts = newRDD.getNumPartitions
    assert(newNParts >= 0)

    val newRangeBounds = Array.range(0, newNParts).map(partitioner.rangeBounds)
    val newPartitioner = partitioner.copy(rangeBounds = newRangeBounds)

    RVD(typ, newPartitioner, newRDD)
  }

  def tail(n: Long, partitionCounts: Option[IndexedSeq[Long]]): RVD = {
    require(n >= 0)

    if (n == 0)
      return RVD.empty(partitioner.sm, typ)

    val (idxFirst, nDrop) = partitionCounts match {
      case Some(pcs) =>
        getPCSubsetOffset(n, pcs.reverseIterator) match {
          case Some(PCSubsetOffset(idx, _, nDrop)) => (pcs.length - idx - 1) -> nDrop
          case None => return this
        }
      case None =>
        val crddCleanup = crdd.cleanupRegions
        val PCSubsetOffset(idx, _, nDrop) =
          incrementalPCSubsetOffset(n, Range.inclusive(getNumPartitions - 1, 0, -1))(
            crddCleanup.runJob(getIteratorSize, _)
          )
        idx -> nDrop
    }
    assert(nDrop < Int.MaxValue)

    val newRDD = crdd.cmapPartitionsAndContextWithIndex(
      { case (i, ctx, f) =>
        val it = f.next()(ctx)
        if (i == idxFirst) {
          (0 until nDrop.toInt).foreach { _ =>
            ctx.region.clear()
            assert(it.hasNext)
            it.next()
          }
          it
        } else
          it
      },
      preservesPartitioning = true,
    )
      .subsetPartitions(Array.range(idxFirst, getNumPartitions))

    val oldNParts = crdd.getNumPartitions
    val newNParts = newRDD.getNumPartitions
    assert(oldNParts >= newNParts)
    assert(newNParts >= 0)

    val newRangeBounds = Array.range(oldNParts - newNParts, oldNParts).map(partitioner.rangeBounds)
    val newPartitioner = partitioner.copy(rangeBounds = newRangeBounds)

    RVD(typ, newPartitioner, newRDD)
  }

  def filter(p: (RVDContext, Long) => Boolean): RVD =
    filterWithContext((_, _) => (), (_: Any, c, l) => p(c, l))

  def filterWithContext[C](makeContext: (Int, RVDContext) => C, f: (C, RVDContext, Long) => Boolean)
    : RVD = {
    val crdd: ContextRDD[Long] =
      this.crdd.cmapPartitionsWithContextAndIndex { (i, consumerCtx, iteratorToFilter) =>
        val c = makeContext(i, consumerCtx)
        val producerCtx = consumerCtx.freshContext
        iteratorToFilter(producerCtx).filter { ptr =>
          val b = f(c, consumerCtx, ptr)
          if (b) {
            producerCtx.region.move(consumerCtx.region)
          } else {
            producerCtx.region.clear()
          }
          b
        }
      }
    RVD(this.typ, this.partitioner, crdd)
  }

  def filterIntervals(intervals: RVDPartitioner, keep: Boolean): RVD =
    if (keep)
      filterToIntervals(intervals)
    else
      filterOutIntervals(intervals)

  def filterOutIntervals(intervals: RVDPartitioner): RVD = {
    val intervalsBc = intervals.broadcast(sparkContext)
    val kType = typ.kType
    val kPType = kType
    val kRowFieldIdx = typ.kFieldIdx
    val rowPType = typ.rowType

    filterWithContext[UnsafeRow](
      (_, _) => new UnsafeRow(kPType),
      { case (kUR, ctx, ptr) =>
        ctx.rvb.start(kType)
        ctx.rvb.selectRegionValue(rowPType, kRowFieldIdx, ctx.r, ptr)
        kUR.set(ctx.region, ctx.rvb.end())
        !intervalsBc.value.contains(kUR)
      },
    )
  }

  def filterToIntervals(intervals: RVDPartitioner): RVD = {
    val intervalsBc = intervals.broadcast(sparkContext)
    val localRowPType = rowPType
    val kRowFieldIdx = typ.kFieldIdx

    val pred: (RVDContext, Long) => Boolean = (ctx: RVDContext, ptr: Long) => {
      val ur = new UnsafeRow(localRowPType, ctx.r, ptr)
      val key = Row.fromSeq(
        kRowFieldIdx.map(i => ur.get(i))
      )
      intervalsBc.value.contains(key)
    }

    val nPartitions = getNumPartitions
    if (nPartitions <= 1)
      return filter(pred)

    val newPartitionIndices = Iterator.range(0, partitioner.numPartitions)
      .filter(i => intervals.overlaps(partitioner.rangeBounds(i)))
      .toArray

    info(s"reading ${newPartitionIndices.length} of $nPartitions data partitions")

    if (newPartitionIndices.isEmpty)
      RVD.empty(intervals.sm, typ)
    else {
      subsetPartitions(newPartitionIndices).filter(pred)
    }
  }

  def subsetPartitions(keep: Array[Int]): RVD = {
    require(keep.length <= crdd.partitions.length, "tried to subset to more partitions than exist")
    require(
      keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < crdd.partitions.length)),
      "values not increasing or not in range [0, number of partitions)",
    )

    val newPartitioner = partitioner.copy(rangeBounds = keep.map(partitioner.rangeBounds))

    RVD(typ, newPartitioner, crdd.subsetPartitions(keep))
  }

  def combine[U: ClassTag, T: ClassTag](
    execCtx: ExecuteContext,
    mkZero: (HailClassLoader, HailTaskContext) => T,
    itF: (HailClassLoader, Int, RVDContext, Iterator[Long]) => T,
    deserialize: (HailClassLoader, HailTaskContext) => (U => T),
    serialize: (HailClassLoader, HailTaskContext, T) => U,
    combOp: (HailClassLoader, HailTaskContext, T, T) => T,
    commutative: Boolean,
    tree: Boolean,
  ): T = {
    var reduced = crdd.cmapPartitionsWithIndex[U] { (i, ctx, it) =>
      Iterator.single(
        serialize(
          theHailClassLoaderForSparkWorkers,
          SparkTaskContext.get(),
          itF(theHailClassLoaderForSparkWorkers, i, ctx, it),
        )
      )
    }

    if (tree) {
      val depth = treeAggDepth(getNumPartitions, HailContext.get.branchingFactor)
      val scale = math.max(
        math.ceil(math.pow(getNumPartitions, 1.0 / depth)).toInt,
        2,
      )

      var i = 0
      while (reduced.getNumPartitions > scale) {
        val nParts = reduced.getNumPartitions
        val newNParts = nParts / scale
        log.info(s"starting tree aggregate stage $i ($nParts => $newNParts partitions)")
        reduced = reduced
          .mapPartitionsWithIndex { (i, it) =>
            it.map(x => (itemPartition(i, nParts, newNParts), (i, x)))
          }
          .partitionBy(new Partitioner {
            override def getPartition(key: Any): Int = key.asInstanceOf[Int]
            override def numPartitions: Int = newNParts
          })
          .cmapPartitions { (ctx, it) =>
            val hcl = theHailClassLoaderForSparkWorkers
            val htc = SparkTaskContext.get()
            var acc = mkZero(hcl, htc)
            it.foreach { case (_, (_, v)) =>
              acc = combOp(hcl, htc, acc, deserialize(hcl, htc)(v))
            }
            Iterator.single(serialize(hcl, htc, acc))
          }
        i += 1
      }
    }

    val ac = Combiner(
      mkZero(execCtx.theHailClassLoader, execCtx.taskContext),
      (acc1: T, acc2: T) => combOp(execCtx.theHailClassLoader, execCtx.taskContext, acc1, acc2),
      commutative,
      true,
    )
    sparkContext.runJob(
      reduced.run,
      (it: Iterator[U]) => singletonElement(it),
      (i, x: U) => ac.combine(i, deserialize(execCtx.theHailClassLoader, execCtx.taskContext)(x)),
    )
    ac.result()
  }

  def count(): Long =
    crdd.boundary.cmapPartitions { (ctx, it) =>
      var count = 0L
      it.foreach(_ => count += 1)
      Iterator.single(count)
    }.run.fold(0L)(_ + _)

  def countPerPartition(): Array[Long] =
    crdd.boundary.cmapPartitions { (ctx, it) =>
      var count = 0L
      it.foreach(_ => count += 1)
      Iterator.single(count)
    }.collect()

  // Collecting

  def collect(execCtx: ExecuteContext): Array[Row] = {
    val enc = TypedCodecSpec(rowPType, BufferSpec.wireSpec)
    val encodedData = collectAsBytes(execCtx, enc)
    val (pType: PStruct, dec) = enc.buildDecoder(execCtx, rowType)
    execCtx.r.pool.scopedRegion { region =>
      RegionValue.fromBytes(execCtx.theHailClassLoader, dec, region, encodedData.iterator)
        .map { ptr =>
          val row = SafeRow(pType, ptr)
          region.clear()
          row
        }.toArray
    }
  }

  def collectAsBytes(ctx: ExecuteContext, enc: AbstractTypedCodecSpec): Array[Array[Byte]] =
    stabilize(ctx, enc).collect()

  // Persisting

  def cache(ctx: ExecuteContext): RVD = persist(ctx, StorageLevel.MEMORY_ONLY)

  def persist(ctx: ExecuteContext, level: StorageLevel): RVD = {
    val enc = TypedCodecSpec(rowPType, BufferSpec.memorySpec)
    val persistedRDD = stabilize(ctx, enc).persist(level)
    val (newRowPType, iterationRDD) = destabilize(ctx, persistedRDD, enc)

    new RVD(RVDType(newRowPType, typ.key), partitioner, iterationRDD) {
      override def storageLevel: StorageLevel = persistedRDD.getStorageLevel

      override def persist(ctx: ExecuteContext, newLevel: StorageLevel): RVD = {
        if (newLevel == StorageLevel.NONE)
          unpersist()
        else {
          persistedRDD.persist(newLevel)
          this
        }
      }

      override def unpersist(): RVD = {
        persistedRDD.unpersist()
        self
      }
    }
  }

  def unpersist(): RVD = this

  def storageLevel: StorageLevel = StorageLevel.NONE

  def write(
    ctx: ExecuteContext,
    path: String,
    idxRelPath: String,
    stageLocally: Boolean,
    codecSpec: AbstractTypedCodecSpec,
  ): Array[FileWriteMetadata] = {
    val fileData = crdd.writeRows(ctx, path, idxRelPath, typ, stageLocally, codecSpec)
    val spec = MakeRVDSpec(
      codecSpec,
      fileData.map(_.path),
      partitioner,
      IndexSpec.emptyAnnotation(idxRelPath, typ.kType),
    )
    spec.write(ctx.fs, path)
    fileData
  }

  // Joining

  def orderedLeftJoinDistinctAndInsert(
    right: RVD,
    root: String,
  ): RVD = {
    assert(!typ.key.contains(root))

    val rightRowType = right.typ.rowType

    val newRowType = rowPType.appendKey(root, right.typ.valueType.setRequired(false))

    val localRowType = rowPType

    val rightValueIndices = right.typ.valueFieldIdx

    val joiner = { (ctx: RVDContext, it: Iterator[JoinedRegionValue]) =>
      val rvb = ctx.rvb
      val rv = RegionValue()

      it.map { jrv =>
        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight
        rvb.start(newRowType)
        rvb.startStruct()
        rvb.addAllFields(localRowType, lrv)
        if (rrv == null)
          rvb.setMissing()
        else {
          rvb.startStruct()
          rvb.addFields(rightRowType, rrv, rightValueIndices)
          rvb.endStruct()
        }
        rvb.endStruct()
        rv.set(ctx.region, rvb.end())
        rv
      }
    }
    assert(typ.key.length >= right.typ.key.length, s"$typ >= ${right.typ}\n  $this\n  $right")
    orderedLeftJoinDistinct(
      right,
      right.typ.key.length,
      joiner,
      typ.copy(rowType = newRowType),
    )
  }

  def orderedLeftJoinDistinct(
    right: RVD,
    joinKey: Int,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType,
  ): RVD =
    keyBy(joinKey).orderedLeftJoinDistinct(right.keyBy(joinKey), joiner, joinedType)

  def orderedLeftIntervalJoin(
    ctx: ExecuteContext,
    right: RVD,
    joiner: PStruct => (
      RVDType,
      (RVDContext, Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => Iterator[RegionValue],
    ),
  ): RVD =
    keyBy(1).orderedLeftIntervalJoin(ctx, right.keyBy(1), joiner)

  def orderedLeftIntervalJoinDistinct(
    ctx: ExecuteContext,
    right: RVD,
    joiner: PStruct => (RVDType, (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue]),
  ): RVD =
    keyBy(1).orderedLeftIntervalJoinDistinct(ctx, right.keyBy(1), joiner)

  def orderedMerge(
    right: RVD,
    joinKey: Int,
    ctx: ExecuteContext,
  ): RVD =
    keyBy(joinKey).orderedMerge(right.keyBy(joinKey), ctx)

  // Zipping

  // New key type must be prefix of left key type. 'joinKey' must be prefix of
  // both left key and right key. 'zipper' must take all output key values from
  // left iterator, and be monotonic on left iterator (it can drop or duplicate
  // elements of left iterator, or insert new elements in order, but cannot
  // rearrange them), and output region values must conform to 'newTyp'. The
  // partitioner of the resulting RVD will be left partitioner truncated
  // to new key. Each partition will be computed by 'zipper', with corresponding
  // partition of 'this' as first iterator, and with all rows of 'that' whose
  // 'joinKey' might match something in partition as the second iterator.
  def alignAndZipPartitions(
    newTyp: RVDType,
    that: RVD,
    joinKey: Int,
  )(
    zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf this.typ.kType)
    require(joinKey <= this.typ.key.length)
    require(joinKey <= that.typ.key.length)

    val sm = partitioner.sm
    val left = this.truncateKey(newTyp.key)
    RVD(
      typ = newTyp,
      partitioner = left.partitioner,
      crdd = left.crdd.toCRDDRegionValue.czipPartitions(
        RepartitionedOrderedRDD2(
          sm,
          that,
          this.partitioner.coarsenedRangeBounds(joinKey),
        ).toCRDDRegionValue
      )(zipper).toCRDDPtr,
    )
  }

  // Like alignAndZipPartitions, when 'that' is keyed by intervals.
  // 'zipper' is called once for each partition of 'this', as in
  // alignAndZipPartitions, but now the second iterator will contain all rows
  // of 'that' whose key is an interval overlapping the range bounds of the
  // current partition of 'this'.
  def intervalAlignAndZipPartitions(
    ctx: ExecuteContext,
    that: RVD,
  )(
    zipper: PStruct => (
      RVDType,
      (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue],
    )
  ): RVD = {
    require(that.rowType.field(that.typ.key(0)).typ.asInstanceOf[
      TInterval
    ].pointType == rowType.field(typ.key(0)).typ)

    val partBc = partitioner.broadcast(sparkContext)
    val rightTyp = that.typ
    val codecSpec = TypedCodecSpec(that.rowPType, BufferSpec.wireSpec)
    val makeEnc = codecSpec.buildEncoder(ctx, that.rowPType)
    val sm = ctx.stateManager
    val partitionKeyedIntervals = that.crdd.cmapPartitions { (ctx, it) =>
      val encoder = new ByteArrayEncoder(theHailClassLoaderForSparkWorkers, makeEnc)
      TaskContext.get.addTaskCompletionListener[Unit](_ => encoder.close())
      it.flatMap { ptr =>
        val r = SafeRow(rightTyp.rowType, ptr)
        val interval = r.getAs[Interval](rightTyp.kFieldIdx(0))
        if (interval != null) {
          val wrappedInterval = interval.copy(
            start = Row(interval.start),
            end = Row(interval.end),
          )
          val bytes = encoder.regionValueToBytes(ptr)
          partBc.value.queryInterval(wrappedInterval).map(i => ((i, interval), bytes))
        } else
          Iterator()
      }
    }.run

    val nParts = getNumPartitions
    val intervalOrd =
      rightTyp.kType.types(0).virtualType.ordering(sm).toOrdering.asInstanceOf[Ordering[Interval]]
    val sorted: RDD[((Int, Interval), Array[Byte])] = new ShuffledRDD(
      partitionKeyedIntervals,
      new Partitioner {
        def getPartition(key: Any): Int = key.asInstanceOf[(Int, Interval)]._1

        def numPartitions: Int = nParts
      },
    ).setKeyOrdering(Ordering.by[(Int, Interval), Interval](_._2)(intervalOrd))

    val (rightPType: PStruct, rightCRDD) = codecSpec.decodeRDD(ctx, that.rowType, sorted.values)
    val (newTyp, f) = zipper(rightPType)
    RVD(
      typ = newTyp,
      partitioner = partitioner,
      crdd = crdd.toCRDDRegionValue.czipPartitions(rightCRDD.toCRDDRegionValue)(f).toCRDDPtr,
    )
  }

  // Private

  private[rvd] def copy(
    typ: RVDType = typ,
    partitioner: RVDPartitioner = partitioner,
    crdd: ContextRDD[Long] = crdd,
  ): RVD =
    RVD(typ, partitioner, crdd)

  private[rvd] def destabilize(
    ctx: ExecuteContext,
    stable: RDD[Array[Byte]],
    enc: AbstractTypedCodecSpec,
  ): (PStruct, ContextRDD[Long]) = {
    val (rowPType: PStruct, dec) = enc.buildDecoder(ctx, rowType)
    (
      rowPType,
      ContextRDD.weaken(stable).cmapPartitions { (ctx, it) =>
        RegionValue.fromBytes(theHailClassLoaderForSparkWorkers, dec, ctx.region, it)
      },
    )
  }

  private[rvd] def crddBoundary: ContextRDD[Long] =
    crdd.boundary

  private[rvd] def keyBy(key: Int = typ.key.length): KeyedRVD =
    new KeyedRVD(this, key)
}

object RVD {
  def empty(ctx: ExecuteContext, typ: RVDType): RVD =
    RVD.empty(ctx.stateManager, typ)

  def empty(sm: HailStateManager, typ: RVDType): RVD =
    RVD(typ, RVDPartitioner.empty(sm, typ.kType.virtualType), ContextRDD.empty[Long]())

  def unkeyed(rowType: PStruct, crdd: ContextRDD[Long]): RVD =
    new RVD(
      RVDType(rowType, FastSeq()),
      RVDPartitioner.unkeyed(null, crdd.getNumPartitions),
      crdd,
    )

  def getKeys(
    ctx: ExecuteContext,
    typ: RVDType,
    crdd: ContextRDD[Long],
  ): ContextRDD[Long] = {
    // The region values in 'crdd' are of type `typ.rowType`
    val localType = typ
    val sm = ctx.stateManager
    crdd.cmapPartitionsWithContext { (consumerCtx, it) =>
      val producerCtx = consumerCtx.freshContext
      val wrv = WritableRegionValue(sm, localType.kType, consumerCtx.region)
      it(producerCtx).map { ptr =>
        wrv.setSelect(localType.rowType, localType.kFieldIdx, ptr, deepCopy = true)
        producerCtx.region.clear()
        wrv.value.offset
      }
    }
  }

  def getKeyInfo(
    ctx: ExecuteContext,
    typ: RVDType,
    // 'partitionKey' is used to check whether the rows are ordered by the first
    // 'partitionKey' key fields, even if they aren't ordered by the full key.
    partitionKey: Int,
    keys: ContextRDD[Long],
  ): Array[RVDPartitionInfo] = {
    // the region values in 'keys' are of typ `typ.keyType`
    val nPartitions = keys.getNumPartitions
    if (nPartitions == 0)
      return Array()

    val rng = new java.util.Random(1)
    val partitionSeed = Array.fill[Long](nPartitions)(rng.nextLong())

    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val localType = typ

    val sm = ctx.stateManager
    val keyInfo = keys.crunJobWithIndex { (i, rvdContext, it) =>
      if (it.hasNext)
        Some(RVDPartitionInfo(
          sm,
          localType,
          partitionKey,
          samplesPerPartition,
          i,
          it,
          partitionSeed(i),
          rvdContext,
        ))
      else
        None
    }.flatten

    val kOrd = PartitionBoundOrdering(sm, typ.kType.virtualType).toOrdering
    keyInfo.sortBy(_.min)(kOrd)
  }

  def coerce(
    execCtx: ExecuteContext,
    typ: RVDType,
    crdd: ContextRDD[Long],
  ): RVD = coerce(execCtx, typ, typ.key.length, crdd)

  def coerce(
    execCtx: ExecuteContext,
    typ: RVDType,
    crdd: ContextRDD[Long],
    fastKeys: ContextRDD[Long],
  ): RVD = coerce(execCtx, typ, typ.key.length, crdd, fastKeys)

  def coerce(
    execCtx: ExecuteContext,
    typ: RVDType,
    partitionKey: Int,
    crdd: ContextRDD[Long],
  ): RVD = {
    val keys = getKeys(execCtx, typ, crdd)
    makeCoercer(execCtx, typ, partitionKey, keys).coerce(typ, crdd)
  }

  def coerce(
    execCtx: ExecuteContext,
    typ: RVDType,
    partitionKey: Int,
    crdd: ContextRDD[Long],
    keys: ContextRDD[Long],
  ): RVD =
    makeCoercer(execCtx, typ, partitionKey, keys).coerce(typ, crdd)

  def makeCoercer(
    execCtx: ExecuteContext,
    fullType: RVDType,
    // keys: RDD[RegionValue[fullType.kType]]
    keys: ContextRDD[Long],
  ): RVDCoercer = makeCoercer(execCtx, fullType, fullType.key.length, keys)

  def makeCoercer(
    execCtx: ExecuteContext,
    fullType: RVDType,
    partitionKey: Int,
    // keys: RDD[RegionValue[fullType.kType]]
    keys: ContextRDD[Long],
  ): RVDCoercer = {
    type CRDD = ContextRDD[Long]

    val unkeyedCoercer: RVDCoercer = new RVDCoercer(fullType) {
      def _coerce(typ: RVDType, crdd: CRDD): RVD = {
        assert(typ.key.isEmpty)
        unkeyed(typ.rowType, crdd)
      }
    }

    if (fullType.key.isEmpty)
      return unkeyedCoercer

    val emptyCoercer: RVDCoercer = new RVDCoercer(fullType) {
      def _coerce(typ: RVDType, crdd: CRDD): RVD = empty(execCtx, typ)
    }

    val keyInfo = getKeyInfo(execCtx, fullType, partitionKey, keys)

    if (keyInfo.isEmpty)
      return emptyCoercer

    val bounds = keyInfo.map(_.interval).toFastSeq
    val pkBounds = bounds.map(_.coarsen(partitionKey))

    def orderPartitions = { crdd: CRDD =>
      val pids = keyInfo.map(_.partitionIndex)
      if (pids.isSorted && crdd.getNumPartitions == pids.length) {
        assert(pids.isEmpty || pids.last < crdd.getNumPartitions)
        crdd
      } else {
        assert(pids.isEmpty || pids.max < crdd.getNumPartitions)
        if (!pids.isSorted)
          info("Coerced dataset with out-of-order partitions.")
        crdd.reorderPartitions(pids)
      }
    }

    val minInfo = keyInfo.minBy(_.sortedness)
    val intraPartitionSortedness = minInfo.sortedness
    val contextStr = minInfo.contextStr

    if (
      intraPartitionSortedness == RVDPartitionInfo.KSORTED
      && RVDPartitioner.isValid(execCtx.stateManager, fullType.kType.virtualType, bounds)
    ) {

      info("Coerced sorted dataset")

      new RVDCoercer(fullType) {
        val unfixedPartitioner =
          new RVDPartitioner(execCtx.stateManager, fullType.kType.virtualType, bounds)
        val newPartitioner = RVDPartitioner.generate(
          execCtx.stateManager,
          fullType.key.take(partitionKey),
          fullType.kType.virtualType,
          bounds,
        )

        def _coerce(typ: RVDType, crdd: CRDD): RVD =
          RVD(typ, unfixedPartitioner, orderPartitions(crdd))
            .repartition(execCtx, newPartitioner, shuffle = false)
      }

    } else if (
      intraPartitionSortedness >= RVDPartitionInfo.TSORTED
      && RVDPartitioner.isValid(
        execCtx.stateManager,
        fullType.kType.virtualType.truncate(partitionKey),
        pkBounds,
      )
    ) {

      info(s"Coerced almost-sorted dataset")
      log.info(s"Unsorted keys: $contextStr")

      new RVDCoercer(fullType) {
        val unfixedPartitioner = new RVDPartitioner(
          execCtx.stateManager,
          fullType.kType.virtualType.truncate(partitionKey),
          pkBounds,
        )
        val newPartitioner = RVDPartitioner.generate(
          execCtx.stateManager,
          fullType.key.take(partitionKey),
          fullType.kType.virtualType.truncate(partitionKey),
          pkBounds,
        )

        def _coerce(typ: RVDType, crdd: CRDD): RVD = {
          RVD(
            typ.copy(key = typ.key.take(partitionKey)),
            unfixedPartitioner,
            orderPartitions(crdd),
          ).repartition(execCtx, newPartitioner, shuffle = false)
            .localSort(typ.key)
        }
      }

    } else {

      info(s"Ordering unsorted dataset with network shuffle")
      log.info(s"Unsorted keys: $contextStr")

      new RVDCoercer(fullType) {
        val newPartitioner =
          calculateKeyRanges(execCtx, fullType, keyInfo, keys.getNumPartitions, partitionKey)

        def _coerce(typ: RVDType, crdd: CRDD): RVD =
          RVD.unkeyed(typ.rowType, crdd)
            .repartition(execCtx, newPartitioner, shuffle = true, filter = false)
      }
    }
  }

  def calculateKeyRanges(
    ctx: ExecuteContext,
    typ: RVDType,
    pInfo: Array[RVDPartitionInfo],
    nPartitions: Int,
    partitionKey: Int,
  ): RVDPartitioner = {
    assert(nPartitions > 0)
    assert(pInfo.nonEmpty)

    val kord = PartitionBoundOrdering(ctx, typ.kType.virtualType).toOrdering
    val min = pInfo.map(_.min).min(kord)
    val max = pInfo.map(_.max).max(kord)
    val samples = pInfo.flatMap(_.samples)

    RVDPartitioner.fromKeySamples(ctx, typ, min, max, samples, nPartitions, partitionKey)
  }

  def apply(
    typ: RVDType,
    partitioner: RVDPartitioner,
    crdd: ContextRDD[Long],
  ): RVD =
    if (!HailContext.get.checkRVDKeys)
      new RVD(typ, partitioner, crdd)
    else
      new RVD(typ, partitioner, crdd).checkKeyOrdering()

  def unify(execCtx: ExecuteContext, rvds: Seq[RVD]): Seq[RVD] = {
    if (rvds.length == 1 || rvds.forall(_.rowPType == rvds.head.rowPType))
      return rvds

    val sm = execCtx.stateManager
    val unifiedRowPType = InferPType.getCompatiblePType(rvds.map(_.rowPType)).asInstanceOf[PStruct]
    val unifiedKey = rvds.map(_.typ.key).reduce((l, r) => commonPrefix(l, r))
    rvds.map { rvd =>
      val srcRowPType = rvd.rowPType
      val newRVDType = rvd.typ.copy(rowType = unifiedRowPType, key = unifiedKey)
      rvd.map(newRVDType)((ctx, ptr) =>
        unifiedRowPType.copyFromAddress(sm, ctx.r, srcRowPType, ptr, false)
      )
    }
  }

  def union(
    rvds: Seq[RVD],
    joinKey: Int,
    ctx: ExecuteContext,
  ): RVD = rvds match {
    case Seq(x) => x
    case first +: _ =>
      assert(rvds.forall(_.rowPType == first.rowPType))

      if (joinKey == 0) {
        val sc = first.sparkContext
        RVD.unkeyed(first.rowPType, ContextRDD.union(sc, rvds.map(_.crdd)))
      } else
        rvds.toArray.treeReduce(_.orderedMerge(_, joinKey, ctx))
  }

  def union(
    rvds: Seq[RVD],
    ctx: ExecuteContext,
  ): RVD =
    union(rvds, rvds.head.typ.key.length, ctx)

  def writeRowsSplitFiles(
    execCtx: ExecuteContext,
    rvds: IndexedSeq[RVD],
    paths: IndexedSeq[String],
    bufferSpec: BufferSpec,
    stageLocally: Boolean,
  ): Array[Array[FileWriteMetadata]] = {
    val first = rvds.head
    rvds.foreach { rvd =>
      if (rvd.typ != first.typ)
        throw new RuntimeException(s"Type mismatch!\n  head: ${first.typ}\n  altr: ${rvd.typ}")
      if (rvd.partitioner != first.partitioner)
        throw new RuntimeException(
          s"Partitioner mismatch!\n  head:${first.partitioner}\n  altr: ${rvd.partitioner}"
        )
    }

    val sc = SparkBackend.sparkContext("writeRowsSplitFiles")
    val localTmpdir = execCtx.localTmpdir
    val fs = execCtx.fs
    val fsBc = fs.broadcast

    val nRVDs = rvds.length
    val partitioner = first.partitioner
    val partitionerBc = partitioner.broadcast(sc)
    val nPartitions = partitioner.numPartitions

    val localTyp = first.typ
    val fullRowType = first.typ.rowType
    val rowsRVType = MatrixType.getRowType(fullRowType)
    val entriesRVType = MatrixType.getSplitEntriesType(fullRowType)

    val rowsCodecSpec = TypedCodecSpec(rowsRVType, bufferSpec)
    val entriesCodecSpec = TypedCodecSpec(entriesRVType, bufferSpec)
    val rowsIndexSpec = IndexSpec.defaultAnnotation("../../index", localTyp.kType)
    val entriesIndexSpec =
      IndexSpec.defaultAnnotation("../../index", localTyp.kType, withOffsetField = true)
    val makeRowsEnc = rowsCodecSpec.buildEncoder(execCtx, fullRowType)
    val makeEntriesEnc = entriesCodecSpec.buildEncoder(execCtx, fullRowType)
    val _makeIndexWriter =
      IndexWriter.builder(execCtx, localTyp.kType, +PCanonicalStruct("entries_offset" -> PInt64()))
    val makeIndexWriter: (String, RegionPool) => IndexWriter =
      _makeIndexWriter(_, theHailClassLoaderForSparkWorkers, SparkTaskContext.get(), _)

    val partDigits = digitsNeeded(nPartitions)
    for (i <- 0 until nRVDs) {
      val path = paths(i)
      fs.mkDir(path + "/rows/rows/parts")
      fs.mkDir(path + "/entries/rows/parts")
      fs.mkDir(path + "/index")
    }

    val partF = {
      (originIdx: Int, originPartIdx: Int, it: Iterator[RVDContext => Iterator[Long]]) =>
        Iterator.single { ctx: RVDContext =>
          val fullPath = paths(originIdx)
          val fileData = RichContextRDDRegionValue.writeSplitRegion(
            localTmpdir,
            fsBc.value,
            fullPath,
            localTyp,
            singletonElement(it)(ctx),
            originPartIdx,
            ctx,
            partDigits,
            stageLocally,
            makeIndexWriter,
            os => makeRowsEnc(os, theHailClassLoaderForSparkWorkers),
            os => makeEntriesEnc(os, theHailClassLoaderForSparkWorkers),
          )
          Iterator.single((fileData, originIdx))
        }
    }

    val partFilePartitionCounts = execCtx.timer.time("writeOriginUnionRDD")(new ContextRDD(
      new OriginUnionRDD(first.crdd.rdd.sparkContext, rvds.map(_.crdd.rdd), partF)
    )
      .collect())

    val fileDataByOrigin =
      Array.fill[BoxedArrayBuilder[FileWriteMetadata]](nRVDs)(new BoxedArrayBuilder())

    for ((fd, oidx) <- partFilePartitionCounts)
      fileDataByOrigin(oidx) += fd

    val fileData = fileDataByOrigin.map(_.result())

    execCtx.timer.time("writeMetadataInParallel")(
      fileData.zipWithIndex
        .par
        .foreach { case (partFiles, i) =>
          val fs = fsBc.value
          val basePath = paths(i)
          RichContextRDDRegionValue.writeSplitSpecs(
            fs,
            basePath,
            rowsCodecSpec,
            entriesCodecSpec,
            rowsIndexSpec,
            entriesIndexSpec,
            localTyp,
            rowsRVType,
            entriesRVType,
            partFiles.map(_.path),
            partitionerBc.value,
          )
        }
    )

    fileData
  }
}

class BroadcastRVD(backend: SparkBackend, rvd: RVD) extends Serializable {
  private[this] val crdd = rvd.crdd

  private[this] val bcPartitions = backend.broadcast(rvd.crdd.partitions)

  def computePartition(idx: Int, region: Region, partitionRegion: Region): Iterator[Long] = {
    val ctx = new RVDContext(partitionRegion, region)
    val it = crdd.iterator(bcPartitions.value(idx), TaskContext.get())
    it.flatMap(_.apply(ctx))
  }
}
