package is.hail.rvd

import is.hail.annotations._
import is.hail.sparkextras._
import is.hail.utils.fatal
import org.apache.spark.rdd.RDD

import scala.collection.generic.Growable

class KeyedOrderedRVD(val rvd: OrderedRVD, val key: Int) {
  require(key <= rvd.typ.key.length && key >= 0)
  val realType: OrderedRVDType = rvd.typ
  val virtType = new OrderedRVDType(realType.key.take(key), realType.rowType)
  val (kType, _) = rvd.rowType.select(virtType.key)

  private def checkJoinCompatability(right: KeyedOrderedRVD) {
    if (!(kType isIsomorphicTo right.kType))
      fatal(
        s"""Incompatible join keys.  Keys must have same length and types, in order:
           | Left key type: ${ kType.toString }
           | Right key type: ${ right.kType.toString }
         """.stripMargin)
  }

  // 'joinedType.key' must be the join key, followed by the remaining left key,
  // followed by the remaining right key, with possible renaming. 'joiner'
  // must copy these 'joinedType.key' fields from the corresponding fields in
  // the JoinedRegionValue.
  def orderedJoin(
    right: KeyedOrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD = {
    checkJoinCompatability(right)

    val newPartitioner = {
      def leftPart = this.rvd.partitioner.strictify
      def rightPart = right.rvd.partitioner.coarsen(key).extendKey(realType.kType)
      (joinType: @unchecked) match {
        case "left" => leftPart
        case "right" => rightPart
        case "inner" => leftPart.intersect(rightPart)
        case "outer" => OrderedRVDPartitioner.generate(
          realType.kType,
          leftPart.rangeBounds ++ rightPart.rangeBounds)
      }
    }
    val repartitionedLeft = rvd.constrainToOrderedPartitioner(newPartitioner)
    val compute: (OrderedRVIterator, OrderedRVIterator, Iterable[RegionValue] with Growable[RegionValue]) => Iterator[JoinedRegionValue] =
      (joinType: @unchecked) match {
        case "inner" => _.innerJoin(_, _)
        case "left" => _.leftJoin(_, _)
        case "right" => _.rightJoin(_, _)
        case "outer" => _.outerJoin(_, _)
      }
    val lTyp = virtType
    val rTyp = right.virtType
    val rRowType = right.realType.rowType

    repartitionedLeft.alignAndZipPartitions(
      joinedType.copy(key = joinedType.key.take(realType.key.length)),
      right.rvd,
      kType
    ) { (ctx, leftIt, rightIt) =>
      val sideBuffer = ctx.freshContext.region
      joiner(
        ctx,
        compute(
          OrderedRVIterator(lTyp, leftIt, ctx),
          OrderedRVIterator(rTyp, rightIt, ctx),
          new RegionValueArrayBuffer(rRowType, sideBuffer)))
    }.extendKeyPreservesPartitioning(joinedType.key)
  }

  def orderedJoinDistinct(
    right: KeyedOrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD = {
    checkJoinCompatability(right)
    val lTyp = virtType
    val rTyp = right.virtType

    val compute: (OrderedRVIterator, OrderedRVIterator) => Iterator[JoinedRegionValue] =
      (joinType: @unchecked) match {
        case "inner" => _.innerJoinDistinct(_)
        case "left" => _.leftJoinDistinct(_)
      }

    rvd.alignAndZipPartitions(
      joinedType,
      right.rvd,
      kType
    ) { (ctx, leftIt, rightIt) =>
      joiner(
        ctx,
        compute(
          OrderedRVIterator(lTyp, leftIt, ctx),
          OrderedRVIterator(rTyp, rightIt, ctx)))
    }
  }

  def orderedZipJoin(right: KeyedOrderedRVD): ContextRDD[RVDContext, JoinedRegionValue] = {
    checkJoinCompatability(right)
    val ranges = this.rvd.partitioner.coarsenedRangeBounds(key) ++
      right.rvd.partitioner.coarsenedRangeBounds(key)
    val newPartitioner = OrderedRVDPartitioner.generate(virtType.key, kType, ranges)

    val repartitionedLeft = this.rvd.constrainToOrderedPartitioner(newPartitioner)
    val repartitionedRight = right.rvd.constrainToOrderedPartitioner(newPartitioner)

    val leftType = this.virtType
    val rightType = right.virtType
    repartitionedLeft.crddBoundary.czipPartitions(repartitionedRight.crddBoundary)
      { (ctx, leftIt, rightIt) =>
        OrderedRVIterator(leftType, leftIt, ctx)
          .zipJoin(OrderedRVIterator(rightType, rightIt, ctx))
      }
  }

  def orderedMerge(right: KeyedOrderedRVD): OrderedRVD = {
    checkJoinCompatability(right)
    require(this.realType.rowType == right.realType.rowType)

    val ranges = this.rvd.partitioner.coarsenedRangeBounds(key) ++
      right.rvd.partitioner.coarsenedRangeBounds(key)
    val newPartitioner = OrderedRVDPartitioner.generate(virtType.key, kType, ranges)

    val repartitionedLeft =
      this.rvd.constrainToOrderedPartitioner(newPartitioner)
    val lType = this.virtType
    val rType = right.virtType

    repartitionedLeft.alignAndZipPartitions(
      this.virtType,
      right.rvd,
      kType
    ) { (ctx, leftIt, rightIt) =>
      OrderedRVIterator(lType, leftIt, ctx)
        .merge(OrderedRVIterator(rType, rightIt, ctx))
    }
  }
}
