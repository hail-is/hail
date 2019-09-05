package is.hail.rvd

import java.util

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir.PruneDeadFields.isSupertype
import is.hail.expr.types._
import is.hail.expr.types.physical.{PInt64, PStruct, PType}
import is.hail.expr.types.virtual.{TArray, TInt64, TInterval, TStruct}
import is.hail.io.{CodecSpec, CodecSpec2, RichContextRDDRegionValue}
import is.hail.io.index.IndexWriter
import is.hail.sparkextras._
import is.hail.utils._
import org.apache.commons.lang3.StringUtils
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}

import scala.language.existentials
import scala.reflect.ClassTag

abstract class RVDCoercer(val fullType: RVDType) {
  final def coerce(typ: RVDType, crdd: ContextRDD[RVDContext, RegionValue]): RVD = {
    require(isSupertype(typ.rowType.virtualType, fullType.rowType.virtualType))
    require(typ.key.sameElements(fullType.key))
    _coerce(typ, crdd)
  }

  protected def _coerce(typ: RVDType, crdd: ContextRDD[RVDContext, RegionValue]): RVD
}

class RVD(
  val typ: RVDType,
  val partitioner: RVDPartitioner,
  val crdd: ContextRDD[RVDContext, RegionValue]
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

  // Change the row type to a physically equivalent type.
  def changeType(newRowType: PStruct): RVD = {
    require(typ.key.forall(newRowType.fieldNames.contains))
    copy(typ = typ.copy(rowType = newRowType))
  }

  // Exporting

  def toRows: RDD[Row] = {
    val localRowType = rowPType
    map(rv => SafeRow(localRowType, rv.region, rv.offset))
  }

  def toUnsafeRows: RDD[UnsafeRow] = {
    val localRowPType = rowPType
    map(rv => new UnsafeRow(localRowPType, rv.region, rv.offset))
  }

  def stabilize(enc: CodecSpec2): RDD[Array[Byte]] = {
    val encoder = enc.buildEncoder(rowPType)
    crdd.mapPartitions(_.map(_.toBytes(encoder))).clearingRun
  }

  def encodedRDD(enc: CodecSpec2): RDD[Array[Byte]] =
    stabilize(enc)

  def keyedEncodedRDD(enc: CodecSpec2, key: IndexedSeq[String] = typ.key): RDD[(Any, Array[Byte])] = {
    val encoder = enc.buildEncoder(rowPType)
    val kFieldIdx = typ.copy(key = key).kFieldIdx

    val localRowPType = rowPType
    crdd.mapPartitions { it =>
      it.map { rv =>
        val keys: Any = SafeRow.selectFields(localRowPType, rv)(kFieldIdx)
        val bytes = rv.toBytes(encoder)
        (keys, bytes)
      }
    }.clearingRun
  }

  // Return an OrderedRVD whose key equals or at least starts with 'newKey'.
  def enforceKey(newKey: IndexedSeq[String], isSorted: Boolean = false): RVD = {
    require(newKey.forall(rowType.hasField))
    val nPreservedFields = typ.key.zip(newKey).takeWhile { case (l, r) => l == r }.length
    require(!isSorted || nPreservedFields > 0 || newKey.isEmpty)

    if (nPreservedFields == newKey.length)
      this
    else if (isSorted)
      truncateKey(newKey.take(nPreservedFields)).extendKeyPreservesPartitioning(newKey).checkKeyOrdering()
    else
      changeKey(newKey)
  }

  // Key and partitioner manipulation
  def changeKey(newKey: IndexedSeq[String]): RVD =
    changeKey(newKey, newKey.length)

  def changeKey(newKey: IndexedSeq[String], partitionKey: Int): RVD =
    RVD.coerce(typ.copy(key = newKey), partitionKey, this.crdd)

  def extendKeyPreservesPartitioning(newKey: IndexedSeq[String]): RVD = {
    require(newKey startsWith typ.key)
    require(newKey.forall(typ.rowType.fieldNames.contains))
    val rvdType = typ.copy(key = newKey)
    if (RVDPartitioner.isValid(rvdType.kType.virtualType, partitioner.rangeBounds))
      copy(typ = rvdType, partitioner = partitioner.copy(kType = rvdType.kType.virtualType))
    else {
      val adjustedPartitioner = partitioner.strictify
      repartition(adjustedPartitioner)
        .copy(typ = rvdType, partitioner = adjustedPartitioner.copy(kType = rvdType.kType.virtualType))
    }
  }

  def checkKeyOrdering(): RVD = {
    val partitionerBc = partitioner.broadcast(crdd.sparkContext)
    val localType = typ
    val localKPType = typ.kType

    new RVD(
      typ,
      partitioner,
      crdd.cmapPartitionsWithIndex { case (i, ctx, it) =>
        val prevK = WritableRegionValue(localType.kType, ctx.freshRegion)
        val kUR = new UnsafeRow(localKPType)

        new Iterator[RegionValue] {
          var first = true

          def hasNext: Boolean = it.hasNext

          def next(): RegionValue = {
            val rv = it.next()

            if (first)
              first = false
            else {
              if (localType.kRowOrd.gt(prevK.value, rv)) {
                kUR.set(prevK.value)
                val prevKeyString = kUR.toString()

                prevK.setSelect(localType.rowType, localType.kFieldIdx, rv)
                kUR.set(prevK.value)
                val currKeyString = kUR.toString()
                fatal(
                  s"""RVD error! Keys found out of order:
                     |  Current key:  $currKeyString
                     |  Previous key: $prevKeyString
                     |This error can occur after a split_multi if the dataset
                     |contains both multiallelic variants and duplicated loci.
                   """.stripMargin)
              }
            }

            prevK.setSelect(localType.rowType, localType.kFieldIdx, rv)
            kUR.set(prevK.value)

            if (!partitionerBc.value.rangeBounds(i).contains(localType.kType.virtualType.ordering, kUR))
              fatal(
                s"""RVD error! Unexpected key in partition $i
                   |  Range bounds for partition $i: ${ partitionerBc.value.rangeBounds(i) }
                   |  Range of partition IDs for key: [${ partitionerBc.value.lowerBound(kUR) }, ${ partitionerBc.value.upperBound(kUR) })
                   |  Invalid key: ${ kUR.toString() }""".stripMargin)
            rv
          }
        }
      })
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
        partitioner = partitioner.coarsen(newKey.length))
  }

  // WARNING: will drop any data with keys falling outside 'partitioner'.
  def repartition(
    newPartitioner: RVDPartitioner,
    shuffle: Boolean = false,
    filter: Boolean = true
  ): RVD = {
    require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    require(shuffle || newPartitioner.kType.isPrefixOf(typ.kType.virtualType))

    if (shuffle) {
      val newType = typ.copy(key = newPartitioner.kType.fieldNames)

      val localRowPType = rowPType
      val kOrdering = newType.kType.virtualType.ordering

      val partBc = newPartitioner.broadcast(crdd.sparkContext)
      val enc = RVD.wireCodec.makeCodecSpec2(rowPType)

      val filtered: RVD = if (filter) filterWithContext[(UnsafeRow, KeyedRow)]({ case (_, _) =>
        val ur = new UnsafeRow(localRowPType, null, 0)
        val key = new KeyedRow(ur, newType.kFieldIdx)
        (ur, key)
      }, { case ((ur, key), rv) =>
        ur.set(rv)
        partBc.value.contains(key)
      }) else this

      val shuffled: RDD[(Any, Array[Byte])] = new ShuffledRDD(
        filtered.keyedEncodedRDD(enc, newType.key),
        newPartitioner.sparkPartitioner(crdd.sparkContext)
      ).setKeyOrdering(kOrdering.toOrdering)

      val (rType: PStruct, shuffledCRDD) = enc.decodeRDD(localRowPType.virtualType, shuffled.values)

      RVD(RVDType(rType, newType.key), newPartitioner, shuffledCRDD)
    } else {
      if (newPartitioner != partitioner)
        new RVD(
          typ.copy(key = typ.key.take(newPartitioner.kType.size)),
          newPartitioner,
          RepartitionedOrderedRDD2(this, newPartitioner.rangeBounds))
      else
        this
    }
  }

  def naiveCoalesce(maxPartitions: Int): RVD = {
    val n = partitioner.numPartitions
    if (maxPartitions >= n)
      return this

    val newN = maxPartitions
    val newNParts = partition(n, newN)
    assert(newNParts.forall(_ > 0))
    val newPartitioner = partitioner.coalesceRangeBounds(newNParts.scanLeft(-1)(_ + _).tail)
    repartition(newPartitioner)
  }

  def coalesce(maxPartitions: Int, shuffle: Boolean): RVD = {
    require(maxPartitions > 0, "cannot coalesce to nPartitions <= 0")
    val n = crdd.partitions.length
    if (!shuffle && maxPartitions >= n)
      return this

    val newPartitioner = if (shuffle) {
      val enc = RVD.wireCodec.makeCodecSpec2(rowPType)
      val shuffledBytes = stabilize(enc).coalesce(maxPartitions, shuffle = true)
      val (newRowPType, shuffled) = destabilize(shuffledBytes, enc)
      if (typ.key.isEmpty)
        return RVD.unkeyed(newRowPType, shuffled)
      else {
        val newType = RVDType(newRowPType, typ.key)
        val keyInfo = RVD.getKeyInfo(newType, newType.key.length, RVD.getKeys(newType, shuffled))
        if (keyInfo.isEmpty)
          return RVD.empty(sparkContext, typ)
        RVD.calculateKeyRanges(
          newType, keyInfo, shuffled.getNumPartitions, newType.key.length)
      }
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

      partitioner.coalesceRangeBounds(newPartEnd)
    }
    if (newPartitioner.numPartitions< maxPartitions)
      warn(s"coalesced to ${ newPartitioner.numPartitions} " +
        s"${ plural(newPartitioner.numPartitions, "partition") }, less than requested $maxPartitions")

    repartition(newPartitioner, shuffle)
  }

  // Key-wise operations

  def distinctByKey(): RVD = {
    val localType = typ
    repartition(partitioner.strictify)
      .mapPartitions(typ, (ctx, it) =>
        OrderedRVIterator(localType, it, ctx)
          .staircase
          .map(_.value)
      )
  }

  def localSort(newKey: IndexedSeq[String]): RVD = {
    require(newKey startsWith typ.key)
    require(newKey.forall(typ.rowType.fieldNames.contains))
    require(partitioner.satisfiesAllowedOverlap(typ.key.length - 1))

    val localTyp = typ
    val sortedRDD = boundary.crdd.cmapPartitions { (consumerCtx, it) =>
      OrderedRVIterator(localTyp, it, consumerCtx).localKeySort(newKey)
    }

    val newType = typ.copy(key = newKey)
    new RVD(
      newType,
      partitioner.copy(kType = newType.kType.virtualType),
      sortedRDD)
  }

  // Mapping

  // None of the mapping methods may modify values of key fields, so that the
  // partitioner remains valid.

  def map[T](f: (RegionValue) => T)(implicit tct: ClassTag[T]): RDD[T] =
    crdd.map(f).clearingRun

  def map(newTyp: RVDType)(f: (RegionValue) => RegionValue): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.map(f))
  }

  def mapPartitions[T: ClassTag](
    f: (Iterator[RegionValue]) => Iterator[T]
  ): RDD[T] = crdd.mapPartitions(f).clearingRun

  def mapPartitions[T: ClassTag](
    f: (RVDContext, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T] = crdd.cmapPartitions(f).clearingRun

  def mapPartitions(
    newTyp: RVDType
  )(f: (Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.mapPartitions(f))
  }

  def mapPartitions(
    newTyp: RVDType,
    f: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitions(f))
  }

  def mapPartitionsWithIndex[T: ClassTag](
    f: (Int, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T] = crdd.mapPartitionsWithIndex(f).clearingRun

  def mapPartitionsWithIndex[T: ClassTag](
    f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T] = crdd.cmapPartitionsWithIndex(f).clearingRun

  def mapPartitionsWithIndex(
    newTyp: RVDType
  )(f: (Int, Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.mapPartitionsWithIndex(f))
  }

  def mapPartitionsWithIndex(
    newTyp: RVDType,
    f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithIndex(f))
  }

  def mapPartitionsWithIndexAndValue[V](
    newTyp: RVDType,
    values: Array[V],
    f: (Int, RVDContext, V, Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf typ.kType)
    RVD(
      newTyp,
      partitioner.coarsen(newTyp.key.length),
      crdd.cmapPartitionsWithIndexAndValue(values, f))
  }

  def zipWithIndex(name: String, partitionCounts: Option[IndexedSeq[Long]] = None): RVD = {
    assert(!typ.key.contains(name))

    val (newRowPType, ins) = rowPType.unsafeStructInsert(PInt64(), List(name))
    val newRowType = newRowPType

    val a = HailContext.backend.broadcast(partitionCounts.map(_.toArray).getOrElse(countPerPartition()).scanLeft(0L)(_ + _))

    val newCRDD = crdd.cmapPartitionsWithIndex({ (i, ctx, it) =>
      val rv2 = RegionValue()
      val rvb = ctx.rvb
      var index = a.value(i)
      it.map { rv =>
        rvb.start(newRowType)
        ins(rv.region, rv.offset, rvb, () => rvb.addLong(index))
        index += 1
        rv2.set(rvb.region, rvb.end())
        rv2
      }
    }, preservesPartitioning = true)

    RVD(
      typ.copy(rowType = newRowType),
      partitioner,
      crdd = newCRDD
    )
  }

  // Filtering

  def head(n: Long, partitionCounts: Option[IndexedSeq[Long]]): RVD = {
    require(n >= 0)

    if (n == 0)
      return RVD.empty(sparkContext, typ)

    val newRDD = crdd.head(n, partitionCounts)
    val newNParts = newRDD.getNumPartitions
    assert(newNParts >= 0)

    val newRangeBounds = Array.range(0, newNParts).map(partitioner.rangeBounds)
    val newPartitioner = partitioner.copy(rangeBounds = newRangeBounds)

    RVD(typ, newPartitioner, newRDD)
  }

  def filter(p: (RegionValue) => Boolean): RVD =
    RVD(typ, partitioner, crddBoundary.filter(p))

  def filterWithContext[C](makeContext: (Int, RVDContext) => C, f: (C, RegionValue) => Boolean): RVD = {
    mapPartitionsWithIndex(typ, { (i, context, it) =>
      val c = makeContext(i, context)
      it.filter { rv =>
        if (f(c, rv))
          true
        else {
          context.r.clear()
          false
        }
      }
    })
  }

  def filterIntervals(intervals: RVDPartitioner, keep: Boolean): RVD = {
    if (keep)
      filterToIntervals(intervals)
    else
      filterOutIntervals(intervals)
  }

  def filterOutIntervals(intervals: RVDPartitioner): RVD = {
    val intervalsBc = intervals.broadcast(sparkContext)
    val kType = typ.kType
    val kPType = kType
    val kRowFieldIdx = typ.kFieldIdx
    val rowPType = typ.rowType

    mapPartitions(typ, { (ctx, it) =>
      val kUR = new UnsafeRow(kPType)
      it.filter { rv =>
        ctx.rvb.start(kType)
        ctx.rvb.selectRegionValue(rowPType, kRowFieldIdx, rv)
        kUR.set(ctx.region, ctx.rvb.end())
        !intervalsBc.value.contains(kUR)
      }
    })
  }

  def filterToIntervals(intervals: RVDPartitioner): RVD = {
    val intervalsBc = intervals.broadcast(sparkContext)
    val localRowPType = rowPType
    val kRowFieldIdx = typ.kFieldIdx

    val pred: (RegionValue) => Boolean = (rv: RegionValue) => {
      val ur = new UnsafeRow(localRowPType, rv)
      val key = Row.fromSeq(
        kRowFieldIdx.map(i => ur.get(i)))
      intervalsBc.value.contains(key)
    }

    val nPartitions = getNumPartitions
    if (nPartitions <= 1)
      return filter(pred)

    val newPartitionIndices = Iterator.range(0, partitioner.numPartitions)
      .filter(i => intervals.overlaps(partitioner.rangeBounds(i)))
      .toArray

    info(s"reading ${ newPartitionIndices.length } of $nPartitions data partitions")

    if (newPartitionIndices.isEmpty)
      RVD.empty(sparkContext, typ)
    else {
      subsetPartitions(newPartitionIndices).filter(pred)
    }
  }

  def subsetPartitions(keep: Array[Int]): RVD = {
    require(keep.length <= crdd.partitions.length, "tried to subset to more partitions than exist")
    require(keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < crdd.partitions.length)),
      "values not increasing or not in range [0, number of partitions)")

    val newPartitioner = partitioner.copy(rangeBounds = keep.map(partitioner.rangeBounds))

    RVD(typ, newPartitioner, crdd.subsetPartitions(keep))
  }

  // Aggregating
  // used in Interpret by TableAggregate2
  def combine[U: ClassTag](
    zeroValue: U,
    itF: (Int, RVDContext, Iterator[RegionValue]) => U,
    combOp: (U, U) => U,
    commutative: Boolean): U = {
    val reduced = crdd.cmapPartitionsWithIndex[U] { (i, ctx, it) => Iterator.single(itF(i, ctx, it)) }
    val ac = Combiner(zeroValue, combOp, commutative, associative = true)
    sparkContext.runJob(reduced.run, (it: Iterator[U]) => singletonElement(it), ac.combine _)
    ac.result()
  }

  // used in Interpret by TableAggregate, MatrixAggregate
  def aggregateWithPartitionOp[PC, U: ClassTag](
    zeroValue: U,
    makePC: (Int, RVDContext) => PC
  )(seqOp: (PC, U, RegionValue) => Unit,
    combOp: (U, U) => U,
    commutative: Boolean
  ): U = {
    val reduced = crdd.cmapPartitionsWithIndex[U] { (i, ctx, it) =>
      val pc = makePC(i, ctx)
      val comb = zeroValue
      it.foreach { rv =>
        seqOp(pc, comb, rv)
        ctx.region.clear()
      }
      Iterator.single(comb)
    }

    val ac = Combiner(zeroValue, combOp, commutative, associative = true)
    sparkContext.runJob(reduced.run, (it: Iterator[U]) => singletonElement(it), ac.combine _)
    ac.result()
  }

  // only used by MatrixMapCols
  def treeAggregateWithPartitionOp[PC, U: ClassTag](
    zeroValue: U,
    makePC: (Int, RVDContext) => PC
  )(seqOp: (PC, U, RegionValue) => U,
    combOp: (U, U) => U,
    depth: Int = treeAggDepth(HailContext.get, crdd.getNumPartitions)
  ): U = {
    val clearingSeqOp = { (ctx: RVDContext, pc: PC, u: U, rv: RegionValue) =>
      val u2 = seqOp(pc, u, rv)
      ctx.region.clear()
      u2
    }
    crdd.treeAggregateWithPartitionOp(zeroValue, makePC, clearingSeqOp, combOp, depth)
  }

  def forall(p: RegionValue => Boolean): Boolean =
    crdd.map(p).clearingRun.forall(x => x)

  def exists(p: RegionValue => Boolean): Boolean =
    crdd.map(p).clearingRun.exists(x => x)

  def count(): Long =
    crdd.cmapPartitions { (ctx, it) =>
      var count = 0L
      it.foreach { rv =>
        count += 1
        ctx.region.clear()
      }
      Iterator.single(count)
    }.run.fold(0L)(_ + _)

  def countPerPartition(): Array[Long] =
    crdd.cmapPartitions { (ctx, it) =>
      var count = 0L
      it.foreach { rv =>
        count += 1
        ctx.region.clear()
      }
      Iterator.single(count)
    }.collect()

  // Collecting

  def collect(): Array[Row] = {
    val enc = RVD.wireCodec.makeCodecSpec2(rowPType)
    val encodedData = collectAsBytes(enc)
    val (pType: PStruct, dec) = enc.buildDecoder(rowType)
    Region.scoped { region =>
      encodedData.iterator
        .map(RegionValue.fromBytes(dec, region, RegionValue(region)))
        .map { rv =>
          val row = SafeRow(pType, rv)
          region.clear()
          row
        }.toArray
    }
  }

  def collectPerPartition[T: ClassTag](f: (Int, RVDContext, Iterator[RegionValue]) => T): Array[T] =
    crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
      Iterator.single(f(i, ctx, it))
    }.collect()

  def collectAsBytes(enc: CodecSpec2): Array[Array[Byte]] = stabilize(enc).collect()

  // Persisting

  def cache(): RVD = persist(StorageLevel.MEMORY_ONLY)

  def persist(level: StorageLevel): RVD = {
    val enc = RVD.memoryCodec.makeCodecSpec2(rowPType)
    val persistedRDD = stabilize(enc).persist(level)
    val (newRowPType, iterationRDD) = destabilize(persistedRDD, enc)

    new RVD(RVDType(newRowPType, typ.key), partitioner, iterationRDD) {
      override def storageLevel: StorageLevel = persistedRDD.getStorageLevel

      override def persist(newLevel: StorageLevel): RVD = {
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

  def write(path: String, idxRelPath: String, stageLocally: Boolean, codecSpec: CodecSpec): Array[Long] = {
    val (partFiles, partitionCounts) = crdd.writeRows(path, idxRelPath, typ, stageLocally, codecSpec.makeCodecSpec2(rowPType))
    rvdSpec(codecSpec, IndexSpec.emptyAnnotation(idxRelPath, typ.kType.virtualType), partFiles).write(HailContext.sFS, path)
    partitionCounts
  }

  def writeRowsSplit(
    path: String,
    codecSpec: CodecSpec,
    stageLocally: Boolean,
    targetPartitioner: RVDPartitioner
  ): Array[Long] = {
    val fs = HailContext.sFS

    fs.mkDir(path + "/rows/rows/parts")
    fs.mkDir(path + "/entries/rows/parts")
    fs.mkDir(path + "/index")

    val bcFS = HailContext.bcFS
    val nPartitions =
      if (targetPartitioner != null)
        targetPartitioner.numPartitions
      else
        crdd.getNumPartitions
    val d = digitsNeeded(nPartitions)

    val fullRowType = typ.rowType
    val rowsRVType = MatrixType.getRowType(fullRowType)
    val entriesRVType = MatrixType.getSplitEntriesType(fullRowType)

    val rowsCodecSpec = codecSpec.makeCodecSpec2(rowsRVType)
    val entriesCodecSpec = codecSpec.makeCodecSpec2(entriesRVType)
    val makeRowsEnc = rowsCodecSpec.buildEncoder(fullRowType, rowsRVType)
    val makeEntriesEnc = entriesCodecSpec.buildEncoder(fullRowType, entriesRVType)
    val makeIndexWriter = IndexWriter.builder(typ.kType, +PStruct("entries_offset" -> PInt64()))

    val localTyp = typ

    val partFilePartitionCounts: Array[(String, Long)] =
      if (targetPartitioner != null) {
        val nInputParts = partitioner.numPartitions
        val nOutputParts = targetPartitioner.numPartitions

        val inputFirst = new Array[Int](nInputParts)
        val inputLast = new Array[Int](nInputParts)
        val outputFirst = new Array[Int](nInputParts)
        val outputLast = new Array[Int](nInputParts)
        var i = 0
        while (i < nInputParts) {
          outputFirst(i) = -1
          outputLast(i) = -1
          inputFirst(i) = i
          inputLast(i) = i
          i += 1
        }

        var j = 0
        while (j < nOutputParts) {
          var (s, e) = partitioner.intervalRange(targetPartitioner.rangeBounds(j))
          s = math.min(s, nInputParts - 1)
          e = math.min(e, nInputParts - 1)

          if (outputFirst(s) == -1)
            outputFirst(s) = j

          if (outputLast(s) < j)
            outputLast(s) = j

          if (inputLast(s) < e)
            inputLast(s) = e

          j += 1
        }

        val sc = crdd.sparkContext
        val targetPartitionerBc = targetPartitioner.broadcast(sc)
        val localRowPType = rowPType
        val outputFirstBc = sc.broadcast(outputFirst)
        val outputLastBc = sc.broadcast(outputLast)

        crdd.blocked(inputFirst, inputLast)
          .cmapPartitionsWithIndex { (i, ctx, it) =>
            val s = outputFirstBc.value(i)
            if (s == -1)
              Iterator.empty
            else {
              val e = outputLastBc.value(i)

              val fs = bcFS.value
              val bit = it.buffered

              val kOrd = localTyp.kType.virtualType.ordering
              val extractKey: (RegionValue) => Any = (rv: RegionValue) => {
                val ur = new UnsafeRow(localRowPType, rv)
                Row.fromSeq(localTyp.kFieldIdx.map(i => ur.get(i)))
              }

              (s to e).iterator.map { j =>
                val b = targetPartitionerBc.value.rangeBounds(j)

                while (bit.hasNext && b.isAbovePosition(kOrd, extractKey(bit.head)))
                  bit.next()

                assert(
                  !bit.hasNext || {
                    val k = extractKey(bit.head)
                    b.contains(kOrd, k) || b.isBelowPosition(kOrd, k)
                  })

                val it2 = new Iterator[RegionValue] {
                  def hasNext: Boolean = {
                    bit.hasNext && b.contains(kOrd, extractKey(bit.head))
                  }

                  def next(): RegionValue = bit.next()
                }

                val partFileAndCount = RichContextRDDRegionValue.writeSplitRegion(
                  fs,
                  path,
                  localTyp,
                  it2,
                  j,
                  ctx,
                  d,
                  stageLocally,
                  makeIndexWriter,
                  makeRowsEnc,
                  makeEntriesEnc)

                partFileAndCount
              }
            }
          }.collect()
      } else {
        crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
          val fs = bcFS.value
          val partFileAndCount = RichContextRDDRegionValue.writeSplitRegion(
            fs,
            path,
            localTyp,
            it,
            i,
            ctx,
            d,
            stageLocally,
            makeIndexWriter,
            makeRowsEnc,
            makeEntriesEnc)

          Iterator.single(partFileAndCount)
        }.collect()
      }

    val (partFiles, partitionCounts) = partFilePartitionCounts.unzip

    RichContextRDDRegionValue.writeSplitSpecs(fs, path, codecSpec, typ, rowsRVType, entriesRVType, partFiles,
      if (targetPartitioner != null) targetPartitioner else partitioner)

    partitionCounts
  }

  // Joining

  def orderedLeftJoinDistinctAndInsert(
    right: RVD,
    root: String): RVD = {
    assert(!typ.key.contains(root))

    val valueStruct = right.typ.valueType
    val rightRowType = right.typ.rowType

    val newRowType = rowPType.appendKey(root, right.typ.valueType)

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
    assert(typ.key.length >= right.typ.key.length, s"$typ >= ${ right.typ }\n  $this\n  $right")
    orderedJoinDistinct(
      right,
      right.typ.key.length,
      "left",
      joiner,
      typ.copy(rowType = newRowType))
  }

  def orderedJoin(
    right: RVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType
  ): RVD =
    orderedJoin(right, typ.key.length, joinType, joiner, joinedType)

  def orderedJoin(
    right: RVD,
    joinKey: Int,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType
  ): RVD =
    keyBy(joinKey).orderedJoin(right.keyBy(joinKey), joinType, joiner, joinedType)

  def orderedJoinDistinct(
    right: RVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType
  ): RVD =
    orderedJoinDistinct(right, typ.key.length, joinType, joiner, joinedType)

  def orderedJoinDistinct(
    right: RVD,
    joinKey: Int,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType
  ): RVD =
    keyBy(joinKey).orderedJoinDistinct(right.keyBy(joinKey), joinType, joiner, joinedType)

  def orderedLeftIntervalJoin(
    right: RVD,
    joiner: PStruct => (RVDType, (RVDContext, Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => Iterator[RegionValue])
  ): RVD =
    keyBy(1).orderedLeftIntervalJoin(right.keyBy(1), joiner)

  def orderedLeftIntervalJoinDistinct(
    right: RVD,
    joiner: PStruct => (RVDType, (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue])
  ): RVD =
    keyBy(1).orderedLeftIntervalJoinDistinct(right.keyBy(1), joiner)

  def orderedZipJoin(right: RVD): (RVDPartitioner, ContextRDD[RVDContext, JoinedRegionValue]) =
    orderedZipJoin(right, typ.key.length)

  def orderedZipJoin(right: RVD, joinKey: Int): (RVDPartitioner, ContextRDD[RVDContext, JoinedRegionValue]) =
    keyBy(joinKey).orderedZipJoin(right.keyBy(joinKey))

  def orderedZipJoin(
    right: RVD,
    joinKey: Int,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType
  ): RVD = {
    val (joinedPartitioner, jcrdd) = orderedZipJoin(right, joinKey)
    RVD(joinedType, joinedPartitioner, jcrdd.cmapPartitions(joiner))
  }

  def orderedMerge(right: RVD, joinKey: Int): RVD =
    keyBy(joinKey).orderedMerge(right.keyBy(joinKey))

  def partitionSortedUnion(rdd2: RVD): RVD = {
    assert(typ == rdd2.typ)
    assert(partitioner == rdd2.partitioner)

    val localTyp = typ
    zipPartitions(typ, rdd2) { (ctx, it, it2) =>
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

  // Zipping

  def zip(
    newTyp: RVDType,
    that: RVD
  )(zipper: (RVDContext, RegionValue, RegionValue) => RegionValue
  ): RVD = RVD(
    newTyp,
    partitioner,
    boundary.crdd.czip(that.boundary.crdd)(zipper))

  def zipPartitions(
    newTyp: RVDType,
    that: RVD
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = RVD(
    newTyp,
    partitioner,
    boundary.crdd.czipPartitions(that.boundary.crdd)(zipper))

  def zipPartitions[T: ClassTag](
    newTyp: RVDType,
    that: ContextRDD[RVDContext, T]
  )(zipper: (Iterator[RegionValue], Iterator[T]) => Iterator[RegionValue]
  ): RVD = RVD(
    newTyp,
    partitioner,
    boundary.crdd.zipPartitions(that)(zipper))

  def zipPartitionsAndContext(
    newTyp: RVDType,
    that: RVD
  )(zipper: (RVDContext, RVDContext => Iterator[RegionValue], RVDContext => Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = RVD(
    newTyp,
    partitioner,
    crdd.czipPartitionsAndContext(that.crdd) { (ctx, lit, rit) =>
      zipper(ctx, ctx => lit.flatMap(_ (ctx)), ctx => rit.flatMap(_ (ctx)))
    }
  )

  def zipPartitionsWithIndex(
    newTyp: RVDType,
    that: RVD
  )(zipper: (Int, RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = RVD(
    newTyp,
    partitioner,
    boundary.crdd.czipPartitionsWithIndex(that.boundary.crdd)(zipper))

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
    joinKey: Int
  )(zipper: (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue]
  ): RVD = {
    require(newTyp.kType isPrefixOf this.typ.kType)
    require(joinKey <= this.typ.key.length)
    require(joinKey <= that.typ.key.length)

    val left = this.truncateKey(newTyp.key)
    RVD(
      typ = newTyp,
      partitioner = left.partitioner,
      crdd = left.crddBoundary.czipPartitions(
        RepartitionedOrderedRDD2(that, this.partitioner.coarsenedRangeBounds(joinKey)).boundary
      )(zipper))
  }

  // Like alignAndZipPartitions, when 'that' is keyed by intervals.
  // 'zipper' is called once for each partition of 'this', as in
  // alignAndZipPartitions, but now the second iterator will contain all rows
  // of 'that' whose key is an interval overlapping the range bounds of the
  // current partition of 'this'.
  def intervalAlignAndZipPartitions(
    that: RVD
  )(zipper: PStruct => (RVDType, (RVDContext, Iterator[RegionValue], Iterator[RegionValue]) => Iterator[RegionValue])
  ): RVD = {
    require(that.rowType.field(that.typ.key(0)).typ.asInstanceOf[TInterval].pointType == rowType.field(typ.key(0)).typ)

    val partBc = partitioner.broadcast(sparkContext)
    val rightTyp = that.typ
    val codecSpec = RVD.wireCodec.makeCodecSpec2(that.rowPType)
    val makeEnc = codecSpec.buildEncoder(that.rowPType)
    val partitionKeyedIntervals = that.boundary.crdd
      .flatMap { rv =>
        val r = SafeRow(rightTyp.rowType, rv)
        val interval = r.getAs[Interval](rightTyp.kFieldIdx(0))
        if (interval != null) {
          val wrappedInterval = interval.copy(
            start = Row(interval.start),
            end = Row(interval.end))
          val bytes = rv.toBytes(makeEnc)
          partBc.value.queryInterval(wrappedInterval).map(i => ((i, interval), bytes))
        } else
          Iterator()
      }.clearingRun

    val nParts = getNumPartitions
    val intervalOrd = rightTyp.kType.types(0).virtualType.ordering.toOrdering.asInstanceOf[Ordering[Interval]]
    val sorted: RDD[((Int, Interval), Array[Byte])] = new ShuffledRDD(
      partitionKeyedIntervals,
      new Partitioner {
        def getPartition(key: Any): Int = key.asInstanceOf[(Int, Interval)]._1

        def numPartitions: Int = nParts
      }
    ).setKeyOrdering(Ordering.by[(Int, Interval), Interval](_._2)(intervalOrd))

    val (rightPType: PStruct, rightCRDD) = codecSpec.decodeRDD(that.rowType, sorted.values)
    val (newTyp, f) = zipper(rightPType)
    RVD(
      typ = newTyp,
      partitioner = partitioner,
      crdd = crddBoundary.czipPartitions(rightCRDD.boundary)(f))
  }

  // Private

  private[rvd] def copy(
    typ: RVDType = typ,
    partitioner: RVDPartitioner = partitioner,
    crdd: ContextRDD[RVDContext, RegionValue] = crdd
  ): RVD =
    RVD(typ, partitioner, crdd)

  private[rvd] def destabilize(
    stable: RDD[Array[Byte]],
    enc: CodecSpec2
  ): (PStruct, ContextRDD[RVDContext, RegionValue]) = {
    val (rowPType: PStruct, dec) = enc.buildDecoder(rowType)
    (rowPType, ContextRDD.weaken[RVDContext](stable).cmapPartitions { (ctx, it) =>
      val rv = RegionValue(ctx.region)
      it.map(RegionValue.fromBytes(dec, ctx.region, rv))
    })
  }

  private[rvd] def crddBoundary: ContextRDD[RVDContext, RegionValue] =
    crdd.boundary

  private[rvd] def keyBy(key: Int = typ.key.length): KeyedRVD =
    new KeyedRVD(this, key)

  private def rvdSpec(codecSpec: CodecSpec, indexSpec: IndexSpec, partFiles: Array[String]): AbstractRVDSpec =
    IndexedRVDSpec(
      typ,
      codecSpec,
      indexSpec,
      partFiles,
      partitioner)
}

object RVD {
  val memoryCodec = CodecSpec.defaultUncompressed

  val wireCodec = memoryCodec

  def empty(sc: SparkContext, typ: RVDType): RVD = {
    RVD(typ,
      RVDPartitioner.empty(typ),
      ContextRDD.empty[RVDContext, RegionValue](sc))
  }

  def unkeyed(rowType: PStruct, crdd: ContextRDD[RVDContext, RegionValue]): RVD =
    new RVD(
      RVDType(rowType, FastIndexedSeq()),
      RVDPartitioner.unkeyed(crdd.getNumPartitions),
      crdd)

  def getKeys(
    typ: RVDType,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): ContextRDD[RVDContext, RegionValue] = {
    // The region values in 'crdd' are of type `typ.rowType`
    val localType = typ
    crdd.cmapPartitions { (ctx, it) =>
      val wrv = WritableRegionValue(localType.kType, ctx.freshRegion)
      it.map { rv =>
        wrv.setSelect(localType.rowType, localType.kFieldIdx, rv)
        wrv.value
      }
    }
  }

  def getKeyInfo(
    typ: RVDType,
    // 'partitionKey' is used to check whether the rows are ordered by the first
    // 'partitionKey' key fields, even if they aren't ordered by the full key.
    partitionKey: Int,
    keys: ContextRDD[RVDContext, RegionValue]
  ): Array[RVDPartitionInfo] = {
    // the region values in 'keys' are of typ `typ.keyType`
    val nPartitions = keys.getNumPartitions
    if (nPartitions == 0)
      return Array()

    val rng = new java.util.Random(1)
    val partitionSeed = Array.fill[Int](nPartitions)(rng.nextInt())

    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val localType = typ

    val keyInfo = keys.cmapPartitionsWithIndex { (i, ctx, it) =>
      val out = if (it.hasNext)
        Iterator(RVDPartitionInfo(localType, partitionKey, samplesPerPartition, i, it, partitionSeed(i), ctx))
      else
        Iterator()
      out
    }.collect()

    keyInfo.sortBy(_.min)(typ.kType.virtualType.ordering.toOrdering)
  }

  def coerce(
    typ: RVDType,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): RVD = coerce(typ, typ.key.length, crdd)

  def coerce(
    typ: RVDType,
    crdd: ContextRDD[RVDContext, RegionValue],
    fastKeys: ContextRDD[RVDContext, RegionValue]
  ): RVD = coerce(typ, typ.key.length, crdd, fastKeys)

  def coerce(
    typ: RVDType,
    partitionKey: Int,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): RVD = {
    val keys = getKeys(typ, crdd)
    makeCoercer(typ, partitionKey, keys).coerce(typ, crdd)
  }

  def coerce(
    typ: RVDType,
    partitionKey: Int,
    crdd: ContextRDD[RVDContext, RegionValue],
    keys: ContextRDD[RVDContext, RegionValue]
  ): RVD = {
    makeCoercer(typ, partitionKey, keys).coerce(typ, crdd)
  }

  def makeCoercer(
    fullType: RVDType,
    // keys: RDD[RegionValue[fullType.kType]]
    keys: ContextRDD[RVDContext, RegionValue]
  ): RVDCoercer = makeCoercer(fullType, fullType.key.length, keys)

  def makeCoercer(
    fullType: RVDType,
    partitionKey: Int,
    // keys: RDD[RegionValue[fullType.kType]]
    keys: ContextRDD[RVDContext, RegionValue]
  ): RVDCoercer = {
    type CRDD = ContextRDD[RVDContext, RegionValue]
    val sc = keys.sparkContext

    val unkeyedCoercer: RVDCoercer = new RVDCoercer(fullType) {
      def _coerce(typ: RVDType, crdd: CRDD): RVD = {
        assert(typ.key.isEmpty)
        unkeyed(typ.rowType, crdd)
      }
    }

    if (fullType.key.isEmpty)
      return unkeyedCoercer

    val emptyCoercer: RVDCoercer = new RVDCoercer(fullType) {
      def _coerce(typ: RVDType, crdd: CRDD): RVD = empty(sc, typ)
    }

    val keyInfo = getKeyInfo(fullType, partitionKey, keys)

    if (keyInfo.isEmpty)
      return emptyCoercer

    val bounds = keyInfo.map(_.interval).toFastIndexedSeq
    val pkBounds = bounds.map(_.coarsen(partitionKey))

    def orderPartitions = { crdd: CRDD =>
      val pids = keyInfo.map(_.partitionIndex)
      if (pids.isSorted && crdd.getNumPartitions == pids.length) {
        assert(pids.isEmpty || pids.last < crdd.getNumPartitions)
        crdd
      }
      else {
        assert(pids.isEmpty || pids.max < crdd.getNumPartitions)
        if (!pids.isSorted)
          info("Coerced dataset with out-of-order partitions.")
        crdd.reorderPartitions(pids)
      }
    }

    val minInfo = keyInfo.minBy(_.sortedness)
    val intraPartitionSortedness = minInfo.sortedness
    val contextStr = minInfo.contextStr

    if (intraPartitionSortedness == RVDPartitionInfo.KSORTED
      && RVDPartitioner.isValid(fullType.kType.virtualType, bounds)) {

      info("Coerced sorted dataset")

      new RVDCoercer(fullType) {
        val unfixedPartitioner =
          new RVDPartitioner(fullType.kType.virtualType, bounds)
        val newPartitioner = RVDPartitioner.generate(
          fullType.key.take(partitionKey), fullType.kType.virtualType, bounds)

        def _coerce(typ: RVDType, crdd: CRDD): RVD = {
          RVD(typ, unfixedPartitioner, orderPartitions(crdd))
            .repartition(newPartitioner, shuffle = false)
        }
      }

    } else if (intraPartitionSortedness >= RVDPartitionInfo.TSORTED
      && RVDPartitioner.isValid(fullType.kType.virtualType.truncate(partitionKey), pkBounds)) {

      info(s"Coerced almost-sorted dataset")
      log.info(s"Unsorted keys: $contextStr")

      new RVDCoercer(fullType) {
        val unfixedPartitioner = new RVDPartitioner(
          fullType.kType.virtualType.truncate(partitionKey),
          pkBounds
        )
        val newPartitioner = RVDPartitioner.generate(
          fullType.key.take(partitionKey),
          fullType.kType.virtualType.truncate(partitionKey),
          pkBounds
        )

        def _coerce(typ: RVDType, crdd: CRDD): RVD = {
          RVD(
            typ.copy(key = typ.key.take(partitionKey)),
            unfixedPartitioner,
            orderPartitions(crdd)
          ).repartition(newPartitioner, shuffle = false)
            .localSort(typ.key)
        }
      }

    } else {

      info(s"Ordering unsorted dataset with network shuffle")
      log.info(s"Unsorted keys: $contextStr")

      new RVDCoercer(fullType) {
        val newPartitioner =
          calculateKeyRanges(fullType, keyInfo, keys.getNumPartitions, partitionKey)

        def _coerce(typ: RVDType, crdd: CRDD): RVD = {
          RVD.unkeyed(typ.rowType, crdd)
            .repartition(newPartitioner, shuffle = true, filter = false)
        }
      }
    }
  }

  def calculateKeyRanges(
    typ: RVDType,
    pInfo: Array[RVDPartitionInfo],
    nPartitions: Int,
    partitionKey: Int
  ): RVDPartitioner = {
    assert(nPartitions > 0)
    assert(pInfo.nonEmpty)

    val kord = typ.kType.virtualType.ordering.toOrdering
    val min = pInfo.map(_.min).min(kord)
    val max = pInfo.map(_.max).max(kord)
    val samples = pInfo.flatMap(_.samples)

    RVDPartitioner.fromKeySamples(typ, min, max, samples, nPartitions, partitionKey)
  }

  def apply(
    typ: RVDType,
    partitioner: RVDPartitioner,
    crdd: ContextRDD[RVDContext, RegionValue]
  ): RVD = {
    if (!HailContext.get.checkRVDKeys)
      return new RVD(typ, partitioner, crdd)
    else
      return new RVD(typ, partitioner, crdd).checkKeyOrdering()
  }

  def union(rvds: Seq[RVD], joinKey: Int): RVD = rvds match {
    case Seq(x) => x
    case first +: _ =>
      assert(rvds.forall(_.rowPType == first.rowPType))
      if (joinKey == 0) {
        val sc = first.sparkContext
        RVD.unkeyed(first.rowPType, ContextRDD.union(sc, rvds.map(_.crdd)))
      } else
        rvds.toArray.treeReduce(_.orderedMerge(_, joinKey))
  }

  def union(rvds: Seq[RVD]): RVD =
    union(rvds, rvds.head.typ.key.length)

  def writeRowsSplitFiles(
    rvds: IndexedSeq[RVD],
    path: String,
    codecSpec: CodecSpec,
    stageLocally: Boolean
  ): Array[Array[Long]] = {
    val first = rvds.head
    require(rvds.forall(rvd => rvd.typ == first.typ && rvd.partitioner == first.partitioner))

    val sc = HailContext.get.sc
    val fs = HailContext.sFS
    val bcFS = HailContext.bcFS

    val nRVDs = rvds.length
    val partitioner = first.partitioner
    val partitionerBc = partitioner.broadcast(sc)
    val nPartitions = partitioner.numPartitions

    val localTyp = first.typ
    val fullRowType = first.typ.rowType
    val rowsRVType = MatrixType.getRowType(fullRowType)
    val entriesRVType = MatrixType.getSplitEntriesType(fullRowType)

    val rowsCodecSpec = codecSpec.makeCodecSpec2(rowsRVType)
    val entriesCodecSpec = codecSpec.makeCodecSpec2(entriesRVType)
    val makeRowsEnc = rowsCodecSpec.buildEncoder(fullRowType, rowsRVType)
    val makeEntriesEnc = entriesCodecSpec.buildEncoder(fullRowType, entriesRVType)
    val makeIndexWriter = IndexWriter.builder(localTyp.kType, +PStruct("entries_offset" -> PInt64()))

    val partDigits = digitsNeeded(nPartitions)
    val fileDigits = digitsNeeded(rvds.length)
    for (i <- 0 until nRVDs) {
      val s = StringUtils.leftPad(i.toString, fileDigits, '0')
      fs.mkDir(path + s + ".mt" + "/rows/rows/parts")
      fs.mkDir(path + s + ".mt" + "/entries/rows/parts")
      fs.mkDir(path + s + ".mt" + "/index")
    }

    val partF = { (originIdx: Int, originPartIdx: Int, it: Iterator[RVDContext => Iterator[RegionValue]]) =>
      Iterator.single { ctx: RVDContext =>
        val fs = bcFS.value
        val s = StringUtils.leftPad(originIdx.toString, fileDigits, '0')
        val fullPath = path + s + ".mt"
        val (f, rowCount) = RichContextRDDRegionValue.writeSplitRegion(
          fs,
          fullPath,
          localTyp,
          singletonElement(it)(ctx),
          originPartIdx,
          ctx,
          partDigits,
          stageLocally,
          makeIndexWriter,
          makeRowsEnc,
          makeEntriesEnc)
        Iterator.single((f, rowCount, originIdx))
      }
    }

    val partFilePartitionCounts = new ContextRDD(
      new OriginUnionRDD(first.crdd.rdd.sparkContext, rvds.map(_.crdd.rdd), partF),
      first.crdd.mkc)
      .collect()

    val partFilesByOrigin = Array.fill[ArrayBuilder[String]](nRVDs)(new ArrayBuilder())
    val partitionCountsByOrigin = Array.fill[ArrayBuilder[Long]](nRVDs)(new ArrayBuilder())

    for ((f, rowCount, oidx) <- partFilePartitionCounts) {
      partFilesByOrigin(oidx) += f
      partitionCountsByOrigin(oidx) += rowCount
    }

    val partFiles = partFilesByOrigin.map(_.result())
    val partCounts = partitionCountsByOrigin.map(_.result())

    sc.parallelize(partFiles.zipWithIndex, partFiles.length)
      .foreach { case (partFiles, i) =>
        val fs = bcFS.value
        val s = StringUtils.leftPad(i.toString, fileDigits, '0')
        val basePath = path + s + ".mt"
        RichContextRDDRegionValue.writeSplitSpecs(fs, basePath, codecSpec, localTyp, rowsRVType, entriesRVType, partFiles, partitionerBc.value)
      }

    partCounts
  }
}
