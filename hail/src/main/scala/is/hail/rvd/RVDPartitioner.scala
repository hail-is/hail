package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.ir.Literal
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.commons.lang.builder.HashCodeBuilder
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row
import org.apache.spark.{Partitioner, SparkContext}

class RVDPartitioner(
  val kType: TStruct,
  // rangeBounds: Array[Interval[kType]]
  // rangeBounds is interval containing all keys within a partition
  val rangeBounds: Array[Interval],
  allowedOverlap: Int
) {
  // expensive, for debugging
  // assert(rangeBounds.forall(SafeRow.isSafe))

  override def toString: String =
    s"RVDPartitioner($kType, ${rangeBounds.mkString("[", ",\n", "]")})"

  def this(
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval],
    allowedOverlap: Int
  ) = this(kType, rangeBounds.toArray, allowedOverlap)

  def this(
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(kType, rangeBounds.toArray, kType.size)

  def this(
    partitionKey: Array[String],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(kType, rangeBounds.toArray, math.max(partitionKey.length - 1, 0))

  def this(
    partitionKey: Option[Int],
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval]
  ) = this(kType, rangeBounds.toArray, partitionKey.map(_ - 1).getOrElse(kType.size))

  require(rangeBounds.forall { case Interval(l, r, _, _) =>
    kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
  })
  require(allowedOverlap >= 0 && allowedOverlap <= kType.size)
  require(RVDPartitioner.isValid(kType, rangeBounds, allowedOverlap))

  val kord: ExtendedOrdering = PartitionBoundOrdering(kType)
  val intervalKeyLT: (Interval, Any) => Boolean = (i, k) => i.isBelowPosition(kord, k)
  val keyIntervalLT: (Any, Interval) => Boolean = (k, i) => i.isAbovePosition(kord, k)
  val intervalLT: (Interval, Interval) => Boolean = (i1, i2) => i1.isBelow(kord, i2)

  def range: Option[Interval] =
    if (rangeBounds.isEmpty)
      None
    else
      Some(Interval(rangeBounds.head.left, rangeBounds.last.right))

  def satisfiesAllowedOverlap(testAllowedOverlap: Int): Boolean =
    RVDPartitioner.isValid(kType, rangeBounds, testAllowedOverlap)

  def isStrict: Boolean = satisfiesAllowedOverlap(kType.size - 1)

  def numPartitions: Int = rangeBounds.length

  def rangeBoundsType = TArray(TInterval(kType))

  override def equals(other: Any): Boolean = other match {
    case that: RVDPartitioner =>
      this.eq(that) || (this.kType == that.kType && this.rangeBounds.sameElements(that.rangeBounds))
    case _ => false
  }

  override def hashCode: Int = {
    val b = new HashCodeBuilder()
    b.append(kType)
    rangeBounds.foreach(b.append)
    b.toHashCode
  }

  @transient @volatile var partitionerBc: Broadcast[RVDPartitioner] = _

  def broadcast(sc: SparkContext): Broadcast[RVDPartitioner] = {
    if (partitionerBc == null) {
      synchronized {
        if (partitionerBc == null)
          partitionerBc = sc.broadcast(this)
      }
    }
    partitionerBc
  }

  def sparkPartitioner(sc: SparkContext): Partitioner = {
    val selfBc = broadcast(sc)

    new Partitioner {
      def numPartitions: Int = selfBc.value.numPartitions

      def getPartition(key: Any): Int = selfBc.value.lowerBound(key)
    }
  }

  // Key manipulation

  def coarsenedRangeBounds(newKeyLen: Int): Array[Interval] =
    rangeBounds.map(_.coarsen(newKeyLen))

  def coarsen(newKeyLen: Int): RVDPartitioner = {
    if (newKeyLen == kType.size)
      this
    else {
      assert(newKeyLen < kType.size)
      new RVDPartitioner(
        kType.truncate(newKeyLen),
        coarsenedRangeBounds(newKeyLen),
        math.min(allowedOverlap, newKeyLen))
    }
  }

  def strictify: RVDPartitioner = extendKey(kType)

  // Adjusts 'rangeBounds' so that 'satisfiesAllowedOverlap(kType.size - 1)'
  // holds, then changes key type to 'newKType'. If 'newKType' is 'kType', still
  // adjusts 'rangeBounds'.
  def extendKey(newKType: TStruct): RVDPartitioner = {
    require(kType isPrefixOf newKType)
    RVDPartitioner.generate(newKType, rangeBounds)
  }

  def extendKeySamePartitions(newKType: TStruct): RVDPartitioner = {
    require(kType isPrefixOf newKType)
    new RVDPartitioner(
      newKType,
      rangeBounds,
      allowedOverlap)
  }

  // Operators (produce new partitioners)

  def subdivide(
    cutPoints: IndexedSeq[IntervalEndpoint],
    allowedOverlap: Int = kType.size
  ): RVDPartitioner = {
    require(cutPoints.forall { case IntervalEndpoint(row, _) =>
      kType.relaxedTypeCheck(row)
    })
    require(allowedOverlap >= 0 && allowedOverlap <= kType.size)
    require(satisfiesAllowedOverlap(allowedOverlap))

    val eord: ExtendedOrdering = kord.intervalEndpointOrdering
    val scalaEOrd: Ordering[IntervalEndpoint] =
      kord.intervalEndpointOrdering.toOrdering.asInstanceOf[Ordering[IntervalEndpoint]]
    val sorted = cutPoints.map(_.coarsenRight(allowedOverlap + 1)).sorted(scalaEOrd)

    var i = 0
    def firstPast(threshold: IntervalEndpoint, start: Int): Int = {
      val iw = sorted.indexWhere(eord.gt(_, threshold), start)
      if (iw == -1) sorted.length else iw
    }
    val newBounds = rangeBounds.flatMap { interval =>
      val first = firstPast(interval.left, i)
      val last = firstPast(interval.right, first)
      val cuts = sorted.slice(first, last)
      i = last
      for {
        (l, r) <- (interval.left +: cuts) zip (cuts :+ interval.right)
        interval <- Interval.orNone(kord, l, r)
      } yield interval
    }

    new RVDPartitioner(kType, newBounds, allowedOverlap)
  }

  def intersect(other: RVDPartitioner): RVDPartitioner = {
    if (!kType.isIsomorphicTo(other.kType))
      throw new AssertionError(s"key types not isomorphic: $kType, ${other.kType}")

    new RVDPartitioner(kType, Interval.intersection(this.rangeBounds, other.rangeBounds, kord.intervalEndpointOrdering))
  }

  def rename(nameMap: Map[String, String]): RVDPartitioner = new RVDPartitioner(
    kType.rename(nameMap),
    rangeBounds,
    allowedOverlap
  )

  def copy(
    kType: TStruct = kType,
    rangeBounds: IndexedSeq[Interval] = rangeBounds,
    allowedOverlap: Int = allowedOverlap
  ): RVDPartitioner =
    new RVDPartitioner(kType, rangeBounds, allowedOverlap)

  def coalesceRangeBounds(newPartEnd: IndexedSeq[Int]): RVDPartitioner = {
    val newRangeBounds = (-1 +: newPartEnd.init).zip(newPartEnd).map { case (s, e) =>
      rangeBounds(s+1).hull(kord, rangeBounds(e))
    }
    copy(rangeBounds = newRangeBounds)
  }

  // Key queries

  def contains(index: Int, key: Any): Boolean =
    rangeBounds(index).contains(kord, key)

  /** Returns 0 <= i <= numPartitions such that partition i is the first which
    * either contains 'key' or is above 'key', returning numPartitions if 'key'
    * is above all partitions.
    *
    * 'key' may be either a Row or an IntervalEndpoint. In the latter case,
    * returns the ID of the first partition which overlaps the interval with
    * left endpoint 'key' and unbounded right endpoint, or numPartitions if
    * none do.
    */
  def lowerBound(key: Any): Int = rangeBounds.lowerBound(key, intervalKeyLT)

  /** Returns 0 <= i <= numPartitions such that partition i is the first which
    * is above 'key', returning numPartitions if 'key' is above all partitions.
    *
    * 'key' may be either a Row or an IntervalEndpoint. In the latter case,
    * returns the ID of the first partition which is completely above the
    * interval with right endpoint 'key' and unbounded left endpoint, or
    * numPartitions if none are.
    */
  def upperBound(key: Any): Int = rangeBounds.upperBound(key, keyIntervalLT)

  /** Returns (lowerBound, upperBound). Interesting cases are:
    * - partitioner contains 'key':
    *   [lowerBound, upperBound) is the range of partition IDs containing 'key'.
    * - 'key' falls in the gap between two partitions:
    *   lowerBound = upperBound is the ID of the first partition above 'key'.
    * - 'key' is below the first partition (or numPartitions = 0):
    *   lowerBound = upperBound = 0
    * - 'key' is above the last partition:
    *   lowerBound = upperBound = numPartitions
    */
  def keyRange(key: Any): (Int, Int) = rangeBounds.equalRange(key, intervalKeyLT, keyIntervalLT)

  def queryKey(key: Any): Range = {
    val (l, u) = keyRange(key)
    Range(l, u)
  }

  def contains(x: Any): Boolean = rangeBounds.containsOrdered(x, intervalKeyLT, keyIntervalLT)

  // Interval queries

  /** Returns 0 <= i <= numPartitions such that partition i is the first which
    * either overlaps 'query' or is above 'query', returning numPartitions if
    * 'query' is completely above all partitions.
    */
  def lowerBoundInterval(query: Interval): Int = rangeBounds.lowerBound(query, intervalLT)

  /** Returns 0 <= i <= numPartitions such that partition i is the first which
    * is above 'query', returning numPartitions if 'query' is completely above
    * or overlaps all partitions.
    */
  def upperBoundInterval(query: Interval): Int = rangeBounds.upperBound(query, intervalLT)

  /** Returns (lowerBound, upperBound). Interesting cases are:
    * - partitioner overlaps 'query':
    *   [lowerBound, upperBound) is the range of partition IDs overlapping 'query'.
    * - 'query' falls in the gap between two partitions:
    *   lowerBound = upperBound is the ID of the first partition above 'query'.
    * - 'query' is completely below the first partition (or numPartitions = 0):
    *   lowerBound = upperBound = 0
    * - 'query' is completely above the last partition:
    *   lowerBound = upperBound = numPartitions
    */
  def intervalRange(query: Interval): (Int, Int) = rangeBounds.equalRange(query, intervalLT)

  def queryInterval(query: Interval): Range = {
    val (l, u) = intervalRange(query)
    Range(l, u)
  }

  def overlaps(query: Interval): Boolean = rangeBounds.containsOrdered(query, intervalLT)

  def isDisjointFrom(query: Interval): Boolean = !overlaps(query)

  def partitionBoundsIRRepresentation: Literal = {
    Literal(TArray(RVDPartitioner.intervalIRRepresentation(kType)),
      rangeBounds.map(i => RVDPartitioner.intervalToIRRepresentation(i, kType.size)).toFastIndexedSeq)
  }

  def keysIfOneToOne(): Option[IndexedSeq[Row]] = {
    if (kType.size == 0) {
      return None
    }
    val lastType = kType.types.last
    if (lastType != TInt32 && lastType != TInt64) {
      return None
    }
    def singleton(interval: Interval): Option[Row] = {
      val left = interval.left.point.asInstanceOf[Row]
      val leftSign = interval.left.sign
      val right = interval.right.point.asInstanceOf[Row]
      val rightSign = interval.right.sign
      var i = 0
      while (i < kType.types.length - 1) {
        if (i >= left.length ||
          i >= right.length ||
          kType.types(i).ordering.compare(left(i), right(i)) != 0) {
          return None
        }
        i += 1
      }
      if (i >= left.length || i >= right.length) {
        None
      } else {
        val someContainedPoints = (leftSign, rightSign) match {
          case (-1,  1) => Seq(left, right)  // [left, right]
          case (-1, -1) => Seq(left)         // [left, right)
          case ( 1,  1) => Seq(right)        // (left, right]
          case ( 1, -1) => Seq()             // (left, right)
        }

        (someContainedPoints, left.isNullAt(i), right.isNullAt(i), lastType) match {
          case (Seq(), _, _, _) =>
            None
          case (_, false, true, _) =>
            None
          case (_, true, false, _) =>
            None
          case (Seq(somePoint), true, true, _) =>
            Some(somePoint)
          case (Seq(left, right), true, true, _) =>
            Some(left) // both null so last field must be equal
          case (Seq(somePoint), false, false, TInt32) =>
            val diff = left.getInt(i) - right.getInt(i)
            if (diff == 1 || diff == -1) {
              Some(somePoint)
            } else {
              None
            }
          case (Seq(somePoint), false, false, TInt64) =>
            val diff = left.getLong(i) - right.getLong(i)
            if (diff == 1 || diff == -1) {
              Some(somePoint)
            } else {
              None
            }
          case (Seq(lPoint, rPoint), false, false, TInt32) =>
            if (left.getInt(i) == right.getInt(i)) {
              Some(lPoint) // equal so it doesn't matter
            } else {
              None
            }
          case (Seq(lPoint, rPoint), false, false, TInt64) =>
            if (left.getLong(i) == right.getLong(i)) {
              Some(lPoint) // equal so it doesn't matter
            } else {
              None
            }
          case x =>
            throw new AssertionError(s"unexpected case $x")
        }
      }
    }

    anyFailAllFail(rangeBounds.map(singleton))
  }
}

object RVDPartitioner {
  def empty(typ: TStruct): RVDPartitioner = {
    new RVDPartitioner(typ, Array.empty[Interval])
  }

  def unkeyed(numPartitions: Int): RVDPartitioner = {
    new RVDPartitioner(
      TStruct.empty,
      Array.fill(numPartitions)(Interval(Row(), Row(), true, true)),
      0)
  }

  def generate(kType: TStruct, intervals: IndexedSeq[Interval]): RVDPartitioner =
    generate(kType.fieldNames, kType, intervals)

  def generate(
    partitionKey: IndexedSeq[String],
    kType: TStruct,
    intervals: IndexedSeq[Interval]
  ): RVDPartitioner = {
    require(intervals.forall { case Interval(l, r, _, _) =>
      kType.relaxedTypeCheck(l) && kType.relaxedTypeCheck(r)
    })

    val allowedOverlap = math.max(partitionKey.length - 1, 0)
    union(kType, intervals, allowedOverlap).subdivide(intervals.map(_.right), allowedOverlap)
  }

  def union(
    kType: TStruct,
    intervals: IndexedSeq[Interval],
    allowedOverlap: Int
  ): RVDPartitioner = {
    val kord = PartitionBoundOrdering(kType)
    val eord = kord.intervalEndpointOrdering
    val iord = Interval.ordering(kord, startPrimary = true)
    val pk = allowedOverlap + 1
    val rangeBounds: IndexedSeq[Interval] =
      if (intervals.isEmpty)
        intervals
      else {
        val unpruned = intervals.sorted(iord.toOrdering.asInstanceOf[Ordering[Interval]])
        val ab = new BoxedArrayBuilder[Interval](intervals.length)
        var tmp = unpruned(0)
        for (i <- unpruned.tail) {
          if (eord.gteq(tmp.right.coarsenRight(pk), i.left.coarsenLeft(pk)))
            tmp = tmp.hull(kord, i)
          else {
            ab += tmp
            tmp = i
          }
        }
        ab += tmp

        ab.result()
      }

    new RVDPartitioner(kType, rangeBounds, allowedOverlap)
  }

  def fromKeySamples(
    typ: RVDType,
    min: Any,
    max: Any,
    keys: IndexedSeq[Any],
    nPartitions: Int,
    partitionKey: Int
  ): RVDPartitioner = {
    require(nPartitions > 0)
    require(typ.kType.virtualType.relaxedTypeCheck(min))
    require(typ.kType.virtualType.relaxedTypeCheck(max))
    require(keys.forall(typ.kType.virtualType.relaxedTypeCheck))

    val kOrd = PartitionBoundOrdering(typ.kType.virtualType).toOrdering
    val sortedKeys = keys.sorted(kOrd)
    val step = (sortedKeys.length - 1).toDouble / nPartitions
    val partitionEdges = Array.tabulate(nPartitions - 1) { i =>
      IntervalEndpoint(sortedKeys(((i + 1) * step).toInt), 1)
    }.toFastIndexedSeq

    val interval = Interval(min, max, true, true)
    new RVDPartitioner(
      typ.kType.virtualType,
      FastIndexedSeq(interval)
    ).subdivide(partitionEdges, math.max(partitionKey - 1, 0))
  }

  def isValid(kType: TStruct, rangeBounds: IndexedSeq[Interval]): Boolean =
    isValid(kType, rangeBounds, kType.size)

  def isValid(
    kType: TStruct,
    rangeBounds: IndexedSeq[Interval],
    allowedOverlap: Int
  ): Boolean = {
    rangeBounds.isEmpty ||
      rangeBounds.zip(rangeBounds.tail).forall { case (left: Interval, right: Interval) =>
        val r = PartitionBoundOrdering(kType).intervalEndpointOrdering.lteqWithOverlap(allowedOverlap)(left.right, right.left)
        if (!r)
          log.info(s"invalid partitioner: !lteqWithOverlap($allowedOverlap)(${ left }.right, ${ right }.left)")
        r
      }
  }

  def intervalIRRepresentation(ts: TStruct): TStruct = {
    val endpointT = TTuple(ts, TInt32)
    TStruct("left" -> endpointT, "right" -> endpointT, "includesLeft" -> TBoolean, "includesRight" -> TBoolean)
  }

  def intervalToIRRepresentation(interval: Interval, len: Int): Row = {
    def processStruct(r: Row): Row = {
      Row(Row.fromSeq((0 until len).map(i => if (i >= r.length) null else r.get(i))), r.length)
    }

    Row(processStruct(interval.left.point.asInstanceOf[Row]),
      processStruct(interval.right.point.asInstanceOf[Row]),
      interval.left.sign < 0,
      interval.right.sign > 0)
  }
}
