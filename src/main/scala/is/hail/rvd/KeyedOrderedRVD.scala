package is.hail.rvd

import is.hail.annotations._
import is.hail.sparkextras._
import is.hail.utils.fatal
import org.apache.spark.rdd.RDD

import scala.collection.generic.Growable

class KeyedOrderedRVD(val rvd: OrderedRVD, val key: Array[String]) {
  val realType: OrderedRVDType = rvd.typ
  val virtType = new OrderedRVDType(key.take(realType.partitionKey.length), key, realType.rowType)
  val (kType, _) = rvd.rowType.select(key)
  require(kType isPrefixOf rvd.typ.kType)

  private def checkJoinCompatability(right: KeyedOrderedRVD) {
    if (!(kType isIsomorphicTo right.kType))
      fatal(
        s"""Incompatible join keys.  Keys must have same length and types, in order:
           | Left key type: ${ kType.toString }
           | Right key type: ${ right.kType.toString }
         """.stripMargin)
  }

  def orderedJoin(
    right: KeyedOrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD = {
    checkJoinCompatability(right)

    val newPartitioner = (joinType: @unchecked) match {
      case "inner" | "left" => this.rvd.partitioner
      case "outer" | "right" =>
        this.rvd.partitioner.enlargeToRange(right.rvd.partitioner.range)
    }
    val repartitionedLeft = new OrderedRVD(realType, newPartitioner, rvd.crdd)
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
      joinedType,
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
    }
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
    val newPartitioner = rvd.partitioner.enlargeToRange(right.rvd.partitioner.range)

    val leftType = this.virtType
    val rightType = right.virtType
    this.rvd.crddBoundary.czipPartitions(
      new UnpartitionedRVD(
        right.realType.rowType,
        ContextRDD(new RepartitionedOrderedRDD2(right.rvd, newPartitioner.coarsenedPKRangeBounds(key.size)))
      ).crddBoundary
    ) { (ctx, leftIt, rightIt) =>
      OrderedRVIterator(leftType, leftIt, ctx)
        .zipJoin(OrderedRVIterator(rightType, rightIt, ctx))
    }
  }

  def orderedMerge(right: KeyedOrderedRVD): OrderedRVD = {
    checkJoinCompatability(right)
    require(this.typ.rowType == right.typ.rowType)

    val newPartitioner = OrderedRVDPartitioner.mergePartitioners(this.rvd.partitioner, right.rvd.partitioner)
    val repartitionedLeft =
      this.rvd.constrainToOrderedPartitioner(this.typ, newPartitioner)
    val repartitionedRight =
      right.rvd.constrainToOrderedPartitioner(right.typ, newPartitioner)
    val leftType = this.typ
    val rightType = right.typ
    repartitionedLeft.zipPartitions(
      this.typ,
      newPartitioner,
      repartitionedRight,
      preservesPartitioning = true
    ) { (ctx, leftIt, rightIt) =>
      OrderedRVIterator(leftType, leftIt, ctx)
        .merge(OrderedRVIterator(rightType, rightIt, ctx))
    }
  }
}
