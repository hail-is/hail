package is.hail.rvd

import is.hail.annotations._
import is.hail.sparkextras._
import is.hail.utils.fatal
import org.apache.spark.rdd.RDD

import scala.collection.generic.Growable

class KeyedOrderedRVD(val rvd: OrderedRVD, val key: Array[String]) {
  val typ: OrderedRVDType = rvd.typ
  val (kType, _) = rvd.rowType.select(key)
  require(kType isPrefixOf rvd.typ.kType)

  private def checkJoinCompatability(right: KeyedOrderedRVD) {
    if (!(kType isIsomorphicTo kType))
      fatal(
        s"""Incompatible join keys.  Keys must have same length and types, in order:
           | Left key type: ${ kType.toString }
           | Right key type: ${ kType.toString }
         """.stripMargin)
  }

  def orderedJoin(
    right: KeyedOrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD = {
    checkJoinCompatability(right)
    val lTyp = typ
    val rTyp = right.typ

    val newPartitioner = (joinType: @unchecked) match {
      case "inner" | "left" => this.rvd.partitioner
      case "outer" | "right" =>
        this.rvd.partitioner.enlargeToRange(right.rvd.partitioner.range)
    }
    val repartitionedLeft =
      this.rvd.constrainToOrderedPartitioner(this.typ, newPartitioner)
    val repartitionedRight =
      right.rvd.constrainToOrderedPartitioner(right.typ, newPartitioner)
    val compute: (OrderedRVIterator, OrderedRVIterator, Iterable[RegionValue] with Growable[RegionValue]) => Iterator[JoinedRegionValue] =
      (joinType: @unchecked) match {
        case "inner" => _.innerJoin(_, _)
        case "left" => _.leftJoin(_, _)
        case "right" => _.rightJoin(_, _)
        case "outer" => _.outerJoin(_, _)
      }

    repartitionedLeft.zipPartitions(
      joinedType,
      newPartitioner,
      repartitionedRight,
      preservesPartitioning = true
    ) { (ctx, leftIt, rightIt) =>
      val sideBuffer = ctx.freshContext.region
      joiner(
        ctx,
        compute(
          OrderedRVIterator(lTyp, leftIt, ctx),
          OrderedRVIterator(rTyp, rightIt, ctx),
          new RegionValueArrayBuffer(rTyp.rowType, sideBuffer)))
    }
  }

  def orderedJoinDistinct(
    right: KeyedOrderedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: OrderedRVDType
  ): OrderedRVD = {
    checkJoinCompatability(right)
    val rekeyedLTyp = new OrderedRVDType(typ.partitionKey, key, typ.rowType)
    val rekeyedRTyp = new OrderedRVDType(right.typ.partitionKey, right.key, right.typ.rowType)

    val newPartitioner = this.rvd.partitioner
    val repartitionedRight = right.rvd.constrainToOrderedPartitioner(
      right.typ.copy(partitionKey = right.typ.key.take(newPartitioner.partitionKey.length)),
      newPartitioner)
    val compute: (OrderedRVIterator, OrderedRVIterator) => Iterator[JoinedRegionValue] =
      (joinType: @unchecked) match {
        case "inner" => _.innerJoinDistinct(_)
        case "left" => _.leftJoinDistinct(_)
      }

    rvd.zipPartitions(
      joinedType,
      newPartitioner,
      repartitionedRight,
      preservesPartitioning = true
    ) { (ctx, leftIt, rightIt) =>
      joiner(
        ctx,
        compute(
          OrderedRVIterator(rekeyedLTyp, leftIt, ctx),
          OrderedRVIterator(rekeyedRTyp, rightIt, ctx)))
    }
  }

  def orderedZipJoin(right: KeyedOrderedRVD): ContextRDD[RVDContext, JoinedRegionValue] = {
    val newPartitioner = rvd.partitioner.enlargeToRange(right.rvd.partitioner.range)

    val repartitionedLeft = rvd.constrainToOrderedPartitioner(typ, newPartitioner)
    val repartitionedRight = right.rvd.constrainToOrderedPartitioner(right.typ, newPartitioner)

    val leftType = this.typ
    val rightType = right.typ
    repartitionedLeft.zipPartitions(
      repartitionedRight,
      preservesPartitioning = true
    ) { (ctx, leftIt, rightIt) =>
      OrderedRVIterator(leftType, leftIt, ctx)
        .zipJoin(OrderedRVIterator(rightType, rightIt, ctx))
    }
  }
}
