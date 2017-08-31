package is.hail.sparkextras

import is.hail.annotations._
import is.hail.expr.{JSONAnnotationImpex, Parser, TArray, TSet, TStruct, Type}
import is.hail.utils._
import org.apache.spark.{Partition, Partitioner, SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.json4s.JsonAST.{JInt, JObject, JString, JValue}

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

case class PartitionKeyInfo2(
  partIndex: Int,
  sortedness: Int,
  // pk
  min: RegionValue,
  max: RegionValue)

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

  def set(rv: RegionValue): Unit = set(rv.region, rv.offset)

  def set(fromRegion: MemoryBuffer, fromOffset: Long) {
    region.clear()
    rvb.start(t)
    rvb.addRegionValue(t, fromRegion, fromOffset)
    value.offset = rvb.end()
  }
}

object PartitionKeyInfo2 {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply(fullKeyType: TStruct, partIndex: Int, it: Iterator[RegionValue]): PartitionKeyInfo2 = {
    assert(it.hasNext)
    val f0 = it.next()

    val pkType = fullKeyType.fields(0).typ

    val fOrd = fullKeyType.unsafeOrdering(missingGreatest = true)
    val pkOrd = pkType.unsafeOrdering(missingGreatest = true)

    val minF = WritableRegionValue(pkType, f0.region, fullKeyType.loadField(f0.region, f0.offset, 0))
    val maxF = WritableRegionValue(pkType, f0.region, fullKeyType.loadField(f0.region, f0.offset, 0))

    var sortedness = KSORTED

    val prevF = WritableRegionValue(fullKeyType, f0)

    while (it.hasNext) {
      val f = it.next()

      val pkOff = fullKeyType.loadField(f.region, f.offset, 0)
      if (fOrd.lt(f, prevF.value)) {
        if (pkOrd.compare(
          f.region, pkOff,
          prevF.region, fullKeyType.loadField(prevF.region, prevF.offset, 0)) < 0)
          sortedness = UNSORTED
        else
          sortedness = sortedness.min(TSORTED)
      }

      if (pkOrd.compare(f.region, pkOff,
        minF.region, minF.offset) < 0)
        minF.set(f.region, pkOff)
      if (pkOrd.compare(f.region, pkOff,
        maxF.region, maxF.offset) > 0)
        maxF.set(f.region, pkOff)

      prevF.set(f)
    }

    PartitionKeyInfo2(partIndex, sortedness, minF.value, maxF.value)
  }
}

object OrderedRDD2 {
  type CoercionMethod = Int

  final val ORDERED_PARTITIONER: CoercionMethod = 0
  final val AS_IS: CoercionMethod = 1
  final val LOCAL_SORT: CoercionMethod = 2
  final val SHUFFLE: CoercionMethod = 3

  def empty(sc: SparkContext,
    partitionKey: String,
    key: String,
    rowType: TStruct): OrderedRDD2 = {
    val pkField = rowType.field(partitionKey)
    val pkIndex = pkField.index
    val pkType = pkField.typ
    val pkOrd = pkType.unsafeOrdering(missingGreatest = true)

    val kField = rowType.field(key)
    val kIndex = kField.index
    val kType = kField.typ
    val kOrd = kType.unsafeOrdering(missingGreatest = true)

    val fullKeyType = TStruct(
      "pk" -> pkType,
      "k" -> kType)

    OrderedRDD2(partitionKey, key, rowType,
      OrderedPartitioner2.empty(sc, fullKeyType),
      sc.emptyRDD[RegionValue])
  }

  def cast(partitionKey: String,
    key: String,
    rowType: TStruct,
    rdd: RDD[RegionValue]): OrderedRDD2 = {
    if (rdd.partitions.isEmpty)
      OrderedRDD2.empty(rdd.sparkContext, partitionKey, key, rowType)
    else
      rdd match {
        case ordered: OrderedRDD2 => ordered.asInstanceOf[OrderedRDD2]
        case _ =>
          (rdd.partitioner: @unchecked) match {
            case Some(p: OrderedPartitioner2) => OrderedRDD2(partitionKey, key, rowType, p.asInstanceOf[OrderedPartitioner2], rdd)
          }
      }
  }

  def apply(partitionKey: String, key: String, rowType: TStruct,
    rdd: RDD[RegionValue], fastKeys: Option[RDD[RegionValue]], hintPartitioner: Option[OrderedPartitioner2]): OrderedRDD2 = {
    val (_, orderedRDD) = coerce(partitionKey, key, rowType, rdd, fastKeys, hintPartitioner)
    orderedRDD
  }

  /**
    * Precondition: the iterator it is PK-sorted.  We lazily K-sort each block
    * of PK-equivalent elements.
    */
  def localKeySort(rowType: TStruct, pkOrd: Ordering[RegionValue], kOrd: Ordering[RegionValue],
    it: Iterator[RegionValue]): Iterator[RegionValue] = {
    new Iterator[RegionValue] {
      private val bit = it.buffered

      private val q = new mutable.PriorityQueue[RegionValue]()(kOrd.reverse)

      def hasNext: Boolean = bit.hasNext || q.nonEmpty

      def next(): RegionValue = {
        if (q.isEmpty) {
          do {
            val rv = bit.next()
            // FIXME ugh, no good answer here
            q.enqueue(RegionValue(
              rv.region.copy(),
              rv.offset))
          } while (bit.hasNext && pkOrd.compare(q.head, bit.head) == 0)
        }

        val rv = q.dequeue()
        rv
      }
    }
  }

  def coerce(partitionKey: String,
    key: String,
    rowType: TStruct,
    rdd: RDD[RegionValue],
    fastKeys: Option[RDD[RegionValue]] = None,
    hintPartitioner: Option[OrderedPartitioner2] = None): (CoercionMethod, OrderedRDD2) = {
    val sc = rdd.sparkContext

    if (rdd.partitions.isEmpty)
      return (ORDERED_PARTITIONER, empty(sc, partitionKey, key, rowType))

    rdd match {
      case ordd: OrderedRDD2 =>
        return (ORDERED_PARTITIONER, ordd.asInstanceOf[OrderedRDD2])
      case _ =>
    }

    val pkField = rowType.field(partitionKey)
    val pkIndex = pkField.index
    val pkType = pkField.typ
    val pkOrd = pkType.unsafeOrdering(missingGreatest = true)

    val kField = rowType.field(key)
    val kIndex = kField.index
    val kType = kField.typ
    val kOrd = kType.unsafeOrdering(missingGreatest = true)

    val fullKeyType = TStruct(
      "pk" -> pkType,
      "k" -> kType)
    val fOrd = fullKeyType.unsafeOrdering(missingGreatest = true)

    val keys = fastKeys.getOrElse(rdd.mapPartitions { it =>
      var krvb = new RegionValueBuilder(null)
      var krv = RegionValue()

      it.map { rv =>
        krvb.set(rv.region)
        krvb.start(fullKeyType)
        krvb.startStruct()
        krvb.addRegionValue(pkType, rv.region, rowType.loadField(rv.region, rv.offset, pkIndex))
        krvb.addRegionValue(kType, rv.region, rowType.loadField(rv.region, rv.offset, kIndex))
        krvb.endStruct()
        krv.set(rv.region, krvb.end())

        krv
      }
    })

    val keyInfo = keys.mapPartitionsWithIndex { case (i, it) =>
      if (it.hasNext)
        Iterator(PartitionKeyInfo2(fullKeyType, i, it))
      else
        Iterator()
    }.collect()

    log.info(s"Partition summaries: ${ keyInfo.zipWithIndex.map { case (pki, i) => s"i=$i,min=${ pki.min },max=${ pki.max }" }.mkString(";") }")

    if (keyInfo.isEmpty)
      return (AS_IS, empty(sc, partitionKey, key, rowType))

    val sortedKeyInfo = keyInfo.sortBy(_.min)(pkOrd)

    val partitionsSorted = sortedKeyInfo.zip(sortedKeyInfo.tail).forall { case (p, pnext) =>
      val r = pkOrd.lteq(p.max, pnext.min)
      if (!r)
        log.info(s"not sorted: p = $p, pnext = $pnext")
      r
    }

    val pkTTBc = BroadcastTypeTree(sc, pkType)

    val sortedness = sortedKeyInfo.map(_.sortedness).min
    if (partitionsSorted && sortedness >= PartitionKeyInfo.TSORTED) {
      val (adjustedPartitions, rangeBounds, adjSortedness) = rangesAndAdjustments(partitionKey, key, rowType, sortedKeyInfo, sortedness)

      // FIXME clean up
      val rangeBoundsType = TArray(pkType)
      val rangeBoundsRegion = MemoryBuffer()
      val rangeBoundsRVB = new RegionValueBuilder(rangeBoundsRegion)
      rangeBoundsRVB.start(rangeBoundsType)
      rangeBoundsRVB.startArray(rangeBounds.length)
      var i = 0
      while (i < rangeBounds.length) {
        rangeBoundsRVB.addRegionValue(pkType, rangeBounds(i))
        i += 1
      }
      rangeBoundsRVB.endArray()

      val unsafeRangeBounds = UnsafeIndexedSeq(sc, rangeBoundsRegion, rangeBoundsType, rangeBoundsRVB.end(), rangeBounds.length)
      val partitioner = new OrderedPartitioner2(adjustedPartitions.length,
        fullKeyType,
        unsafeRangeBounds)

      val reorderedPartitionsRDD = rdd.reorderPartitions(sortedKeyInfo.map(_.partIndex))
      val adjustedRDD = new AdjustedPartitionsRDD(reorderedPartitionsRDD, adjustedPartitions)
      (adjSortedness: @unchecked) match {
        case PartitionKeyInfo.KSORTED =>
          info("Coerced sorted dataset")
          (AS_IS, OrderedRDD2(partitionKey, key, rowType,
            partitioner,
            adjustedRDD))

        case PartitionKeyInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          (LOCAL_SORT, OrderedRDD2(partitionKey, key, rowType,
            partitioner,
            adjustedRDD.mapPartitions { it =>
              localKeySort(rowType,
                new Ordering[RegionValue] {
                  def compare(rv1: RegionValue, rv2: RegionValue): Int = {
                    pkOrd.compare(rv1.region, rowType.loadField(rv1.region, rv1.offset, pkIndex),
                      rv2.region, rowType.loadField(rv2.region, rv2.offset, pkIndex))
                  }
                },
                new Ordering[RegionValue] {
                  def compare(rv1: RegionValue, rv2: RegionValue): Int = {
                    kOrd.compare(rv1.region, rowType.loadField(rv1.region, rv1.offset, kIndex),
                      rv2.region, rowType.loadField(rv2.region, rv2.offset, kIndex))
                  }
                },
                it)
            }))
      }
    } else {
      info("Ordering unsorted dataset with network shuffle")
      val p = hintPartitioner
        .filter(_.numPartitions >= rdd.partitions.length)
        .getOrElse {
          val ranges: UnsafeIndexedSeq = calculateKeyRanges(fullKeyType, keys)
          new OrderedPartitioner2(ranges.length + 1, fullKeyType, ranges)
        }
      (SHUFFLE, shuffle(partitionKey, key, rowType, p, rdd))
    }
  }

  def calculateKeyRanges(keyType: TStruct, keysRDD: RDD[RegionValue]): UnsafeIndexedSeq = {
    val n = keysRDD.getNumPartitions

    val pkType = keyType.fields(0).typ
    val pkOrd = pkType.unsafeOrdering(missingGreatest = true)

    // FIXME sample up to 1m
    val keys = keysRDD.map { rv =>
      RegionValue(rv.region.copy(), keyType.loadField(rv.region, rv.offset, 0))
    }.collect()
      .sorted(pkOrd)

    val rangeBounds =
      if (keys.length <= n)
        keys.init
      else {
        val k = keys.length / n
        assert(k > 0)
        Array.tabulate(n - 1)(i => keys((i + 1) * k))
      }

    // FIXME clean up
    val rangeBoundsType = TArray(pkType)
    val rangeBoundsRegion = MemoryBuffer()
    val rangeBoundsRVB = new RegionValueBuilder(rangeBoundsRegion)
    rangeBoundsRVB.start(rangeBoundsType)
    rangeBoundsRVB.startArray(rangeBounds.length)
    var i = 0
    while (i < rangeBounds.length) {
      rangeBoundsRVB.addRegionValue(pkType, rangeBounds(i))
      i += 1
    }
    rangeBoundsRVB.endArray()

    val unsafeRangeBounds = UnsafeIndexedSeq(keysRDD.sparkContext, rangeBoundsRegion, rangeBoundsType, rangeBoundsRVB.end(), rangeBounds.length)
    unsafeRangeBounds
  }

  def shuffle(partitionKey: String,
    key: String,
    rowType: TStruct,
    partitioner: OrderedPartitioner2,
    rdd: RDD[RegionValue]): OrderedRDD2 = {
    val sc = rdd.sparkContext

    val pkField = rowType.field(partitionKey)
    val pkIndex = pkField.index
    val pkType = pkField.typ

    val kField = rowType.field(key)
    val kIndex = kField.index
    val kType = kField.typ

    val fullKeyType = TStruct(
      "pk" -> pkType,
      "k" -> kType)

    // FIXME just for assert
    val partitionerBc = sc.broadcast(partitioner)

    val fOrd = fullKeyType.unsafeOrdering(missingGreatest = true)
    OrderedRDD2(partitionKey, key, rowType,
      partitioner,
      // FIXME OrderedPartitioner2 should take fullKey, but bounds should be pk
      new ShuffledRDD[RegionValue, RegionValue, RegionValue](
        rdd.mapPartitions { it =>
          val fRegion = MemoryBuffer()
          val frvb = new RegionValueBuilder(fRegion)
          val frv = RegionValue(fRegion, 0)

          it.map { rv =>
            fRegion.clear()
            frvb.start(fullKeyType)
            frvb.startStruct()
            frvb.addRegionValue(pkType, rv.region, rowType.loadField(rv.region, rv.offset, pkIndex))
            frvb.addRegionValue(kType, rv.region, rowType.loadField(rv.region, rv.offset, kIndex))
            frvb.endStruct()

            frv.offset = frvb.end()

            (frv, rv)
          }
        },
        partitioner)
        .setKeyOrdering(fOrd)
        .mapPartitionsWithIndex { case (i, it) =>
          it.map { case (k, v) =>
            assert(partitionerBc.value.getPartition(k) == i)
            v
          }
        })
  }

  def rangesAndAdjustments(
    partitionKey: String,
    key: String,
    rowType: TStruct,
    sortedKeyInfo: Array[PartitionKeyInfo2],
    sortedness: Int): (IndexedSeq[Array[Adjustment[RegionValue]]], Array[RegionValue], Int) = {

    val pkField = rowType.field(partitionKey)
    val pkIndex = pkField.index
    val pkType = pkField.typ
    val pkOrd = pkType.unsafeOrdering(missingGreatest = true)

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
      while (continue && it.hasNext && pkOrd.equiv(sortedKeyInfo(it.head).min, max)) {
        anyOverlaps = true
        if (pkOrd.equiv(sortedKeyInfo(it.head).max, max))
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
            if (adjustmentsBuffer.nonEmpty && pkOrd.equiv(min, sortedKeyInfo(adjustmentsBuffer.last.head.index).max))
              _.dropWhile(rv => pkOrd.compare(rv.region, rowType.loadField(rv.region, rv.offset, pkIndex),
                min.region, min.offset) == 0)
            else
              identity
          else
          // In every subsequent partition, only take elements that are the max of the last
            _.takeWhile(rv => pkOrd.compare(rv.region, rowType.loadField(rv.region, rv.offset, pkIndex),
              max.region, max.offset) == 0)
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

  def apply(partitionKey: String,
    key: String,
    rowType: TStruct,
    rangeBounds1: Array[Annotation],
    rdd1: RDD[Annotation]): OrderedRDD2 = {
    val sc = rdd1.sparkContext

    val pkField = rowType.field(partitionKey)
    val pkIndex = pkField.index
    val pkType = pkField.typ
    val pkOrd = pkType.unsafeOrdering(missingGreatest = true)

    val kField = rowType.field(key)
    val kIndex = kField.index
    val kType = kField.typ
    val kOrd = kType.unsafeOrdering(missingGreatest = true)

    val fullKeyType = TStruct(
      "pk" -> pkType,
      "k" -> kType)

    val rangeBoundsType = TArray(pkType)
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    rvb.start(rangeBoundsType)
    rvb.startArray(rangeBounds1.length)
    var i = 0
    while (i < rangeBounds1.length) {
      rvb.addAnnotation(pkType, rangeBounds1(i))
      i += 1
    }
    rvb.endArray()
    val aoff = rvb.end()
    val rangeBounds = UnsafeIndexedSeq(sc, region, rangeBoundsType, aoff, rangeBounds1.length)

    val rowTTBc = BroadcastTypeTree(sc, rowType)
    OrderedRDD2(
      partitionKey,
      key,
      rowType,
      new OrderedPartitioner2(rdd1.getNumPartitions,
        fullKeyType,
        rangeBounds),
      rdd1.mapPartitions { it =>
        val rowType = rowTTBc.value.typ
        val region = MemoryBuffer()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region, 0)

        it.map { a =>
          region.clear()
          rvb.start(rowType)
          rvb.addAnnotation(rowType, a)
          rv.offset = rvb.end()

          rv
        }
      })
  }

  def apply(partitionKey: String,
    key: String,
    rowType: TStruct,
    orderedPartitioner: OrderedPartitioner2,
    rdd: RDD[RegionValue]): OrderedRDD2 = {

    val sc = rdd.sparkContext
    val pkField = rowType.field(partitionKey)
    val pkIndex = pkField.index
    val pkType = pkField.typ
    assert(pkType == orderedPartitioner.rangeBoundsType.elementType)

    val partitionerBc = sc.broadcast(orderedPartitioner)

    val rangeBoundsTTBc = BroadcastTypeTree(sc, orderedPartitioner.rangeBoundsType)
    val pkTTBc = BroadcastTypeTree(sc, pkType)

    new OrderedRDD2(partitionKey, key, rowType, orderedPartitioner,
      rdd.mapPartitionsWithIndex { case (i, it) =>
        val partitioner = partitionerBc.value
        val pkOrd = partitioner.unsafeOrdering
        val prevRegion = MemoryBuffer()
        val prevB = new RegionValueBuilder(prevRegion)
        val prev: RegionValue = RegionValue(prevRegion, 0)

        new Iterator[RegionValue] {
          var first = true

          def hasNext: Boolean = it.hasNext

          def next(): RegionValue = {
            val rv = it.next()

            val pkOff = rowType.loadField(rv.region, rv.offset, pkIndex)

            if (i < partitioner.rangeBounds.length)
              assert(pkOrd.compare(rv.region, pkOff,
                partitioner.region, partitioner.loadElement(i)) <= 0)
            if (i > 0)
              assert(pkOrd.compare(partitioner.region, partitioner.loadElement(i - 1),
                rv.region, pkOff) < 0)

            if (first)
              first = false
            else
              assert(pkOrd.compare(prev.region, prev.offset, rv.region, pkOff) <= 0)

            prevRegion.clear()
            prevB.start(pkType)
            prevB.addRegionValue(pkType, rv.region, pkOff)
            prev.offset = prevB.end()

            assert(pkOrd.compare(prev.region, prev.offset, rv.region, pkOff) == 0)

            rv
          }
        }
      })
  }
}

class OrderedRDD2 private(
  val partitionKey: String,
  val key: String,
  val rowType: TStruct,
  @transient val orderedPartitioner: OrderedPartitioner2,
  val rdd: RDD[RegionValue]) extends RDD[RegionValue](rdd) {

  @transient override val partitioner: Option[Partitioner] = Some(orderedPartitioner)

  override def getPartitions: Array[Partition] = rdd.partitions

  override def compute(split: Partition, context: TaskContext): Iterator[RegionValue] = rdd.iterator(split, context)

  override def getPreferredLocations(split: Partition): Seq[String] = rdd.preferredLocations(split)

  def mapPreservesPartitioning(f: (RegionValue) => RegionValue): OrderedRDD2 =
    OrderedRDD2(partitionKey, key, rowType,
      orderedPartitioner,
      rdd.map(f))

  def mapPartitionsPreservesPartitioning(f: (Iterator[RegionValue]) => Iterator[RegionValue]): OrderedRDD2 =
    OrderedRDD2(partitionKey, key, rowType,
      orderedPartitioner,
      rdd.mapPartitions(f))

  override def filter(p: (RegionValue) => Boolean): OrderedRDD2 =
    OrderedRDD2(partitionKey, key, rowType,
      orderedPartitioner,
      rdd.filter(p))
}

object OrderedPartitioner2 {
  def empty(sc: SparkContext, fullKeyType: TStruct): OrderedPartitioner2 = {
    val pkType = fullKeyType.fields(0).typ
    new OrderedPartitioner2(0, fullKeyType, UnsafeIndexedSeq.empty(sc, TArray(pkType)))
  }

  def apply(sc: SparkContext, jv: JValue): OrderedPartitioner2 = {
    case class Extract(numPartitions: Int,
      fullKeyType: String,
      rangeBounds: JValue)
    val ex = jv.extract[Extract]
    val fullKeyType = Parser.parseType(ex.fullKeyType).asInstanceOf[TStruct]
    val pkType = fullKeyType.fields(0).typ
    val rangeBoundsType = TArray(pkType)
    new OrderedPartitioner2(ex.numPartitions,
      fullKeyType,
      UnsafeIndexedSeq(
        sc,
        rangeBoundsType,
        JSONAnnotationImpex.importAnnotation(ex.rangeBounds, rangeBoundsType).asInstanceOf[IndexedSeq[Annotation]]))
  }
}

class OrderedPartitioner2(
  val numPartitions: Int,
  val fullKeyType: TStruct,
  // rangeBounds is partition max
  val rangeBounds: UnsafeIndexedSeq) extends Partitioner {
  require((numPartitions == 0 && rangeBounds.isEmpty) || numPartitions == rangeBounds.length + 1,
    s"nPartitions = $numPartitions, ranges = ${ rangeBounds.length }")
  // require(rangeBounds.isEmpty || rangeBounds.zip(rangeBounds.tail).forall { case (left, right) => left < right })

  val pkType: Type = fullKeyType.fields(0).typ

  val rangeBoundsType = TArray(pkType)
  assert(rangeBoundsType.typeCheck(rangeBounds))

  val ordering: Ordering[Annotation] = pkType.ordering(missingGreatest = true)

  val unsafeOrdering: UnsafeOrdering = pkType.unsafeOrdering(missingGreatest = true)

  def region: MemoryBuffer = rangeBounds.region

  def loadElement(i: Int): Long = rangeBoundsType.loadElement(rangeBounds.region, rangeBounds.aoff, rangeBounds.length, i)

  // return the smallest partition for which key <= max
  // key: fullKeyType
  def getPartition(key: Any): Int = {
    val keyrv = key.asInstanceOf[RegionValue]
    val pkOff = fullKeyType.loadField(keyrv.region, keyrv.offset, 0)

    val part = BinarySearch.binarySearch(numPartitions,
      // elem.compare(key)
      i =>
        if (i == numPartitions - 1)
          -1 // key.compare(inf)
        else
          unsafeOrdering.compare(keyrv.region, pkOff, rangeBounds.region, loadElement(i)))
    part
  }

  def toJSON: JValue =
    JObject(List(
      "numPartitions" -> JInt(numPartitions),
      "fullKeyType" -> JString(fullKeyType.toString),
      "rangeBounds" -> JSONAnnotationImpex.exportAnnotation(rangeBounds, rangeBoundsType)))
}
