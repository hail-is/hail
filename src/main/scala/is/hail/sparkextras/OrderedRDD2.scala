package is.hail.sparkextras

import java.util

import is.hail.annotations._
import is.hail.expr.{JSONAnnotationImpex, Parser, TArray, TStruct, Type}
import is.hail.utils._
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.apache.spark._
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.MappingException
import org.json4s.JsonAST._

import scala.collection.mutable

object BinarySearch {
  // return smallest elem such that key <= elem
  def binarySearch(length: Int,
    // key.compare(elem)
    compare: (Int) => Int): Int = {
    assert(length > 0)

    var low = 0
    var high = length - 1
    while (low < high) {
      val mid = (low + high) / 2
      assert(mid >= low && mid < high)

      // key <= elem
      if (compare(mid) <= 0) {
        high = mid
      } else {
        low = mid + 1
      }
    }
    assert(low == high)
    assert(low >= 0 && low < length)

    // key <= low
    assert(compare(low) <= 0 || low == length - 1)
    // low == 0 || (low - 1) > key
    assert(low == 0
      || compare(low - 1) > 0)

    low
  }
}

object WritableRegionValue {
  def apply(t: Type, initial: RegionValue): WritableRegionValue =
    WritableRegionValue(t, initial.region, initial.offset)

  def apply(t: Type, initialRegion: MemoryBuffer, initialOffset: Long): WritableRegionValue = {
    val wrv = WritableRegionValue(t)
    wrv.set(initialRegion, initialOffset)
    wrv
  }

  def apply(t: Type): WritableRegionValue = {
    val region = MemoryBuffer()
    new WritableRegionValue(t, region, new RegionValueBuilder(region), RegionValue(region, 0))
  }
}

class WritableRegionValue(val t: Type,
  val region: MemoryBuffer,
  rvb: RegionValueBuilder,
  val value: RegionValue) {

  def offset: Long = value.offset

  def setSelect(fromT: TStruct, toFromFieldIdx: Array[Int], fromRV: RegionValue) {
    t match {
      case t: TStruct =>
        region.clear()
        rvb.start(t)
        rvb.startStruct()
        var i = 0
        while (i < t.size) {
          rvb.addField(fromT, fromRV, toFromFieldIdx(i))
          i += 1
        }
        rvb.endStruct()
        value.setOffset(rvb.end())
    }
  }

  def set(rv: RegionValue): Unit = set(rv.region, rv.offset)

  def set(fromRegion: MemoryBuffer, fromOffset: Long) {
    region.clear()
    rvb.start(t)
    rvb.addRegionValue(t, fromRegion, fromOffset)
    value.setOffset(rvb.end())
  }
}

object PartitionKeyInfo2 {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply(typ: OrderedRDD2Type, sampleSize: Int, partitionIndex: Int, it: Iterator[RegionValue], seed: Int): PartitionKeyInfo2 = {
    val minF = WritableRegionValue(typ.pkType)
    val maxF = WritableRegionValue(typ.pkType)
    val prevF = WritableRegionValue(typ.kType)

    assert(it.hasNext)
    val f0 = it.next()

    minF.setSelect(typ.kType, typ.pkKFieldIdx, f0)
    maxF.setSelect(typ.kType, typ.pkKFieldIdx, f0)
    prevF.set(f0)

    var sortedness = KSORTED

    val rng = new java.util.Random(seed)
    val samples = new Array[WritableRegionValue](sampleSize)

    var i = 0
    while (it.hasNext) {
      val f = it.next()

      if (typ.kOrd.compare(f, prevF.value) < 0) {
        if (typ.pkInKOrd.compare(f, prevF.value) < 0)
          sortedness = UNSORTED
        else
          sortedness = sortedness.min(TSORTED)
      }

      if (typ.pkKOrd.compare(minF.value, f) > 0)
        minF.setSelect(typ.kType, typ.pkKFieldIdx, f)
      if (typ.pkKOrd.compare(maxF.value, f) < 0)
        maxF.setSelect(typ.kType, typ.pkKFieldIdx, f)

      prevF.set(f)

      if (i < sampleSize)
        samples(i) = WritableRegionValue(typ.pkType, f)
      else {
        val j = rng.nextInt(i)
        if (j < sampleSize)
          samples(j).set(f)
      }

      i += 1
    }

    PartitionKeyInfo2(partitionIndex, i,
      minF.value, maxF.value,
      Array.tabulate[RegionValue](math.min(i, sampleSize))(i => samples(i).value),
      sortedness)
  }
}

case class PartitionKeyInfo2(
  partitionIndex: Int,
  size: Int,
  min: RegionValue,
  max: RegionValue,
  // min, max: RegionValue[pkType]
  samples: Array[RegionValue],
  sortedness: Int)

object OrderedRDD2Type {
  def selectUnsafeOrdering(t1: TStruct, fields1: Array[Int],
    t2: TStruct, fields2: Array[Int]): UnsafeOrdering = {
    require(fields1.length == fields2.length)
    require((fields1, fields2).zipped.forall { case (f1, f2) =>
      t1.fieldType(f1) == t2.fieldType(f2)
    })

    val nFields = fields1.length
    val fieldOrderings = fields1.map(f1 => t1.fieldType(f1).unsafeOrdering(missingGreatest = true))

    new UnsafeOrdering {
      def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
        var i = 0
        while (i < nFields) {
          val f1 = fields1(i)
          val f2 = fields2(i)
          val leftDefined = t1.isFieldDefined(r1, o1, f1)
          val rightDefined = t2.isFieldDefined(r2, o2, f2)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, t1.loadField(r1, o1, f1), r2, t2.loadField(r2, o2, f2))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            return c
          }

          i += 1
        }

        0
      }
    }
  }

  def apply(jv: JValue): OrderedRDD2Type = {
    case class Extract(partitionKey: Array[String],
      key: Array[String],
      rowType: String)
    val ex = jv.extract[Extract]
    new OrderedRDD2Type(ex.partitionKey, ex.key, Parser.parseType(ex.rowType).asInstanceOf[TStruct])
  }
}

class OrderedRDD2Type(
  val partitionKey: Array[String],
  val key: Array[String], // full key
  val rowType: TStruct) extends Serializable {
  assert(key.startsWith(partitionKey))

  val (pkType, _) = rowType.select(partitionKey)
  val (kType, _) = rowType.select(key)

  val keySet: Set[String] = key.toSet
  val (valueType, _) = rowType.filter(f => !keySet.contains(f.name))

  val valueFieldIdx: Array[Int] = (0 until rowType.size)
    .filter(i => !keySet.contains(rowType.fields(i).name))
    .toArray

  val kRowFieldIdx: Array[Int] = key.map(n => rowType.fieldIdx(n))
  val pkRowFieldIdx: Array[Int] = partitionKey.map(n => rowType.fieldIdx(n))
  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))
  assert(pkKFieldIdx sameElements (0 until pkType.size))

  val pkOrd: UnsafeOrdering = pkType.unsafeOrdering(missingGreatest = true)
  val kOrd: UnsafeOrdering = kType.unsafeOrdering(missingGreatest = true)

  val pkRowOrd: UnsafeOrdering = OrderedRDD2Type.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, rowType, pkRowFieldIdx)
  val pkKOrd: UnsafeOrdering = OrderedRDD2Type.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, kType, pkKFieldIdx)
  val pkInRowOrd: UnsafeOrdering = OrderedRDD2Type.selectUnsafeOrdering(rowType, pkRowFieldIdx, rowType, pkRowFieldIdx)
  val kInRowOrd: UnsafeOrdering = OrderedRDD2Type.selectUnsafeOrdering(rowType, kRowFieldIdx, rowType, kRowFieldIdx)
  val pkInKOrd: UnsafeOrdering = OrderedRDD2Type.selectUnsafeOrdering(kType, pkKFieldIdx, kType, pkKFieldIdx)

  def toJSON: JValue =
    JObject(List(
      "partitionKey" -> JArray(partitionKey.map(JString).toList),
      "key" -> JArray(key.map(JString).toList),
      "rowType" -> JString(rowType.toString)))

  override def equals(that: Any): Boolean = that match {
    case that: OrderedRDD2Type =>
      (partitionKey sameElements that.partitionKey) &&
        (key sameElements that.key) &&
        rowType == that.rowType
    case _ => false
  }

  override def hashCode: Int = {
    val b = new HashCodeBuilder(43, 19)
    b.append(partitionKey.length)
    partitionKey.foreach(b.append)

    b.append(key.length)
    key.foreach(b.append)

    b.append(rowType)
    b.toHashCode
  }
}

object OrderedRDD2 {
  type CoercionMethod = Int

  final val ORDERED_PARTITIONER: CoercionMethod = 0
  final val AS_IS: CoercionMethod = 1
  final val LOCAL_SORT: CoercionMethod = 2
  final val SHUFFLE: CoercionMethod = 3

  def empty(sc: SparkContext, typ: OrderedRDD2Type): OrderedRDD2 = {
    OrderedRDD2(typ,
      OrderedPartitioner2.empty(typ),
      sc.emptyRDD[RegionValue])
  }

  def cast(typ: OrderedRDD2Type,
    rdd: RDD[RegionValue]): OrderedRDD2 = {
    if (rdd.partitions.isEmpty)
      OrderedRDD2.empty(rdd.sparkContext, typ)
    else
      rdd match {
        case ordered: OrderedRDD2 => ordered.asInstanceOf[OrderedRDD2]
        case _ =>
          (rdd.partitioner: @unchecked) match {
            case Some(p: OrderedPartitioner2) => OrderedRDD2(typ, p.asInstanceOf[OrderedPartitioner2], rdd)
          }
      }
  }

  def apply(typ: OrderedRDD2Type,
    rdd: RDD[RegionValue], fastKeys: Option[RDD[RegionValue]], hintPartitioner: Option[OrderedPartitioner2]): OrderedRDD2 = {
    val (_, orderedRDD) = coerce(typ, rdd, fastKeys, hintPartitioner)
    orderedRDD
  }

  /**
    * Precondition: the iterator it is PK-sorted.  We lazily K-sort each block
    * of PK-equivalent elements.
    */
  def localKeySort(typ: OrderedRDD2Type,
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
  def getKeys(typ: OrderedRDD2Type,
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

  def getPartitionKeyInfo(typ: OrderedRDD2Type,
    // keys: RDD[kType]
    keys: RDD[RegionValue]): Array[PartitionKeyInfo2] = {
    val nPartitions = keys.getNumPartitions

    val rng = new java.util.Random(1)
    val partitionSeed = Array.tabulate[Int](nPartitions)(i => rng.nextInt())

    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val pkis = keys.mapPartitionsWithIndex { case (i, it) =>
      if (it.hasNext)
        Iterator(PartitionKeyInfo2(typ, samplesPerPartition, i, it, partitionSeed(i)))
      else
        Iterator()
    }.collect()

    pkis.sortBy(_.min)(typ.pkOrd)
  }

  def coerce(typ: OrderedRDD2Type,
    // rdd: RDD[RegionValue[rowType]]
    rdd: RDD[RegionValue],
    // fastKeys: Option[RDD[RegionValue[kType]]]
    fastKeys: Option[RDD[RegionValue]] = None,
    hintPartitioner: Option[OrderedPartitioner2] = None): (CoercionMethod, OrderedRDD2) = {
    val sc = rdd.sparkContext

    if (rdd.partitions.isEmpty)
      return (ORDERED_PARTITIONER, empty(sc, typ))

    rdd match {
      case ordd: OrderedRDD2 =>
        return (ORDERED_PARTITIONER, ordd.asInstanceOf[OrderedRDD2])
      case _ =>
    }

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
      val partitioner = new OrderedPartitioner2(adjustedPartitions.length,
        typ.partitionKey,
        typ.kType,
        unsafeRangeBounds)

      val reorderedPartitionsRDD = rdd.reorderPartitions(pkis.map(_.partitionIndex))
      val adjustedRDD = new AdjustedPartitionsRDD(reorderedPartitionsRDD, adjustedPartitions)
      (adjSortedness: @unchecked) match {
        case PartitionKeyInfo.KSORTED =>
          info("Coerced sorted dataset")
          (AS_IS, OrderedRDD2(typ,
            partitioner,
            adjustedRDD))

        case PartitionKeyInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          (LOCAL_SORT, OrderedRDD2(typ,
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
          new OrderedPartitioner2(ranges.length + 1, typ.partitionKey, typ.kType, ranges)
        }
      (SHUFFLE, shuffle(typ, p, rdd))
    }
  }

  def calculateKeyRanges(typ: OrderedRDD2Type, pkis: Array[PartitionKeyInfo2], nPartitions: Int): UnsafeIndexedSeq = {
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

  def shuffle(typ: OrderedRDD2Type,
    partitioner: OrderedPartitioner2,
    rdd: RDD[RegionValue]): OrderedRDD2 = {
    OrderedRDD2(typ,
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

  def rangesAndAdjustments(typ: OrderedRDD2Type,
    sortedKeyInfo: Array[PartitionKeyInfo2],
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

  def apply(typ: OrderedRDD2Type,
    partitioner: OrderedPartitioner2,
    rdd: RDD[RegionValue]): OrderedRDD2 = {
    val sc = rdd.sparkContext

    new OrderedRDD2(typ, partitioner,
      rdd.mapPartitionsWithIndex { case (i, it) =>
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

  def leftJoin(left: OrderedRDD2Type, right: OrderedRDD2Type): (
    TStruct, UnsafeOrdering, (RegionValueBuilder, RegionValue, RegionValue, RegionValue) => Unit) = {
    if (left.kType != right.kType)
      fatal(
        s"""Incompatible join keys.  Keys must have same length and types, in order:
           | Left key type: ${ left.kType.toPrettyString(compact = true) }
           | Right key type: ${ right.kType.toPrettyString(compact = true) }
         """.stripMargin)

    // checks disjoint names
    val joinType = left.rowType ++ right.valueType

    val lrKOrd = OrderedRDD2Type.selectUnsafeOrdering(left.rowType, left.kRowFieldIdx,
      right.rowType, right.kRowFieldIdx)

    val merge = { (rvb: RegionValueBuilder, jrv: RegionValue, lrv: RegionValue, rrv: RegionValue) =>
      rvb.set(lrv.region)
      rvb.start(joinType)
      rvb.startStruct() // row

      var i = 0
      while (i < left.rowType.size) {
        rvb.addField(left.rowType, lrv, i)
        i += 1
      }

      if (rrv != null) {
        i = 0
        while (i < right.valueFieldIdx.length) {
          rvb.addField(right.rowType, rrv, right.valueFieldIdx(i))
          i += 1
        }
      } else {
        i = 0
        while (i < right.valueFieldIdx.length) {
          rvb.setMissing()
          i += 1
        }
      }
      rvb.endStruct()

      // leave result in lrv
      jrv.set(lrv.region, rvb.end())
    }

    (joinType, lrKOrd, merge)
  }
}

class OrderedRDD2 private(
  val typ: OrderedRDD2Type,
  @transient val orderedPartitioner: OrderedPartitioner2,
  val rdd: RDD[RegionValue]) extends RDD[RegionValue](rdd) {

  def this(partitionKey: String, key: String, rowType: TStruct,
    orderedPartitioner: OrderedPartitioner2,
    rdd: RDD[RegionValue]) = this(new OrderedRDD2Type(Array(partitionKey), Array(partitionKey, key), rowType),
    orderedPartitioner, rdd)

  @transient override val partitioner: Option[Partitioner] = Some(orderedPartitioner)

  override def getPartitions: Array[Partition] = rdd.partitions

  override def compute(split: Partition, context: TaskContext): Iterator[RegionValue] = rdd.iterator(split, context)

  override def getPreferredLocations(split: Partition): Seq[String] = rdd.preferredLocations(split)

  def mapPreservesPartitioning(f: (RegionValue) => RegionValue): OrderedRDD2 =
    OrderedRDD2(typ,
      orderedPartitioner,
      rdd.map(f))

  def mapPartitionsPreservesPartitioning(f: (Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRDD2 =
    OrderedRDD2(typ,
      orderedPartitioner,
      rdd.mapPartitions(f))

  override def filter(p: (RegionValue) => Boolean): OrderedRDD2 =
    OrderedRDD2(typ,
      orderedPartitioner,
      rdd.filter(p))

  override def sample(withReplacement: Boolean, fraction: Double, seed: Long): OrderedRDD2 =
    OrderedRDD2(typ,
      orderedPartitioner,
      rdd.sample(withReplacement, fraction, seed))

  def getStorageLevel2: StorageLevel = StorageLevel.NONE

  def unpersist2(): OrderedRDD2 = throw new IllegalArgumentException("not persisted")

  def persist2(level: StorageLevel): OrderedRDD2 = {
    val localRowType = typ.rowType

    // copy, persist region values
    val persistedRDD = rdd.mapPartitions { it =>
      val region = MemoryBuffer()
      val rvb = new RegionValueBuilder(region)
      it.map { rv =>
        region.clear()
        rvb.start(localRowType)
        rvb.addRegionValue(localRowType, rv)
        val off = rvb.end()
        RegionValue(region.copy(), off)
      }
    }
      .persist(level)

    val rdd2 = persistedRDD
      .mapPartitions { it =>
        val region = MemoryBuffer()
        val rv2 = RegionValue(region)
        it.map { rv =>
          region.setFrom(rv.region)
          rv2.setOffset(rv.offset)
          rv2
        }
      }

    val self = this

    new OrderedRDD2(typ,
      orderedPartitioner,
      rdd2) {
      override def getStorageLevel2: StorageLevel = level

      override def persist2(newLevel: StorageLevel): OrderedRDD2 = {
        if (newLevel == level)
          this
        else
          self.persist2(newLevel)
      }

      override def unpersist2(): OrderedRDD2 = {
        persistedRDD.unpersist()
        self
      }
    }
  }

  def orderedLeftJoinDistinct(right: OrderedRDD2): OrderedRDD2 = {
    val (joinType, lrKOrd, merge) = OrderedRDD2.leftJoin(typ, right.typ)

    OrderedRDD2(new OrderedRDD2Type(typ.partitionKey, typ.key, joinType),
      orderedPartitioner,
      new OrderedLeftJoinDistinctRDD2(this, right, lrKOrd, merge))
  }

  def partitionSortedUnion(rdd2: OrderedRDD2): OrderedRDD2 = {
    assert(typ == rdd2.typ)
    assert(orderedPartitioner == rdd2.orderedPartitioner)

    val localTyp = typ
    OrderedRDD2(typ, orderedPartitioner,
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
              val c = typ.kInRowOrd.compare(bit.head, bit2.head)
              if (c < 0)
                bit.next()
              else
                bit2.next()
            }
          }
        }
      })
  }
}

class OrderedDependency2(left: OrderedRDD2, right: OrderedRDD2) extends NarrowDependency[RegionValue](right) {
  override def getParents(partitionId: Int): Seq[Int] =
    OrderedDependency2.getDependencies(left.orderedPartitioner, right.orderedPartitioner)(partitionId)
}

object OrderedDependency2 {
  def getDependencies(p1: OrderedPartitioner2, p2: OrderedPartitioner2)(partitionId: Int): Range = {
    val lastPartition = if (partitionId == p1.rangeBounds.length)
      p2.numPartitions - 1
    else
      p2.getPartitionPK(p1.rangeBounds(partitionId))

    if (partitionId == 0)
      0 to lastPartition
    else {
      val startPartition = p2.getPartitionPK(p1.rangeBounds(partitionId - 1))
      startPartition to lastPartition
    }
  }
}

case class OrderedLeftJoinDistinctRDD2Partition(index: Int, leftPartition: Partition, rightPartitions: Array[Partition]) extends Partition

class OrderedLeftJoinDistinctRDD2(left: OrderedRDD2, right: OrderedRDD2,
  lrKOrd: UnsafeOrdering,
  merge: (RegionValueBuilder, RegionValue, RegionValue, RegionValue) => Unit)
  extends RDD[RegionValue](left.sparkContext,
    Seq[Dependency[_]](new OneToOneDependency(left),
      new OrderedDependency2(left, right))) {

  override val partitioner: Option[Partitioner] = left.partitioner

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](left.getNumPartitions)(i =>
      OrderedLeftJoinDistinctRDD2Partition(i,
        left.partitions(i),
        OrderedDependency2.getDependencies(left.orderedPartitioner, right.orderedPartitioner)(i)
          .map(right.partitions)
          .toArray))
  }

  override def getPreferredLocations(split: Partition): Seq[String] = left.preferredLocations(split)

  override def compute(split: Partition, context: TaskContext): Iterator[RegionValue] = {
    val partition = split.asInstanceOf[OrderedLeftJoinDistinctRDD2Partition]

    val leftIt = left.iterator(partition.leftPartition, context)
    val rightIt = partition.rightPartitions.iterator.flatMap { p =>
      right.iterator(p, context)
    }

    new Iterator[RegionValue] {
      val rvb = new RegionValueBuilder()
      val jrv = RegionValue()

      var rrv: RegionValue =
        if (rightIt.hasNext)
          rightIt.next()
        else
          null

      def advanceRight(lrv: RegionValue) {
        while (rrv != null && lrKOrd.compare(lrv, rrv) > 0) {
          if (rightIt.hasNext)
            rrv = rightIt.next()
          else
            null
        }
      }

      def hasNext: Boolean = leftIt.hasNext

      def next(): RegionValue = {
        val lrv = leftIt.next()
        advanceRight(lrv)
        // merge into lrv
        merge(rvb, jrv, lrv,
          // FIXME duplicate comparison
          if (rrv != null && lrKOrd.compare(lrv, rrv) == 0)
            rrv
          else
            null)
        jrv
      }
    }
  }
}

object OrderedPartitioner2 {
  def empty(typ: OrderedRDD2Type): OrderedPartitioner2 = {
    new OrderedPartitioner2(0, typ.partitionKey, typ.kType, UnsafeIndexedSeq.empty(TArray(typ.pkType)))
  }

  def apply(sc: SparkContext, jv: JValue): OrderedPartitioner2 = {
    case class Extract(numPartitions: Int,
      partitionKey: Array[String],
      kType: String,
      rangeBounds: JValue)
    val ex = jv.extract[Extract]

    val partitionKey = ex.partitionKey
    val kType = Parser.parseType(ex.kType).asInstanceOf[TStruct]
    val (pkType, _) = kType.select(partitionKey)

    val rangeBoundsType = TArray(pkType)
    new OrderedPartitioner2(ex.numPartitions,
      ex.partitionKey,
      kType,
      UnsafeIndexedSeq(
        rangeBoundsType,
        JSONAnnotationImpex.importAnnotation(ex.rangeBounds, rangeBoundsType).asInstanceOf[IndexedSeq[Annotation]]))
  }
}

class OrderedPartitioner2(
  val numPartitions: Int,
  val partitionKey: Array[String], val kType: TStruct,
  // rangeBounds is partition max, sorted ascending
  // rangeBounds: Array[pkType]
  val rangeBounds: UnsafeIndexedSeq) extends Partitioner {
  require((numPartitions == 0 && rangeBounds.isEmpty) || numPartitions == rangeBounds.length + 1,
    s"nPartitions = $numPartitions, ranges = ${ rangeBounds.length }")

  val (pkType, _) = kType.select(partitionKey)

  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))
  val pkKOrd: UnsafeOrdering = OrderedRDD2Type.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, kType, pkKFieldIdx)

  val rangeBoundsType = TArray(pkType)
  assert(rangeBoundsType.typeCheck(rangeBounds))

  val ordering: Ordering[Annotation] = pkType.ordering(missingGreatest = true)
  require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left, right) => ordering.compare(left, right) < 0 })

  def region: MemoryBuffer = rangeBounds.region

  def loadElement(i: Int): Long = rangeBoundsType.loadElement(rangeBounds.region, rangeBounds.aoff, rangeBounds.length, i)

  // return the smallest partition for which key <= max
  // pk: Annotation[pkType]
  def getPartitionPK(pk: Any): Int = {
    assert(pkType.typeCheck(pk))

    val part = BinarySearch.binarySearch(numPartitions,
      // key.compare(elem)
      i =>
        if (i == numPartitions - 1)
          -1 // key.compare(inf)
        else
          ordering.compare(pk, rangeBounds(i)))
    part
  }

  // return the smallest partition for which key <= max
  // key: RegionValue[kType]
  def getPartition(key: Any): Int = {
    val keyrv = key.asInstanceOf[RegionValue]

    val part = BinarySearch.binarySearch(numPartitions,
      // key.compare(elem)
      i =>
        if (i == numPartitions - 1)
          -1 // key.compare(inf)
        else
          -pkKOrd.compare(rangeBounds.region, loadElement(i), keyrv))
    part
  }

  def toJSON: JValue =
    JObject(List(
      "numPartitions" -> JInt(numPartitions),
      "partitionKey" -> JArray(partitionKey.map(n => JString(n)).toList),
      "kType" -> JString(kType.toPrettyString(compact = true)),
      "rangeBounds" -> JSONAnnotationImpex.exportAnnotation(rangeBounds, rangeBoundsType)))
}
