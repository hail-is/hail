package is.hail.rvd

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.types.physical.PStruct
import is.hail.types.virtual.TInterval
import is.hail.sparkextras._
import is.hail.utils._

import scala.collection.generic.Growable

class KeyedRVD(val rvd: RVD, val key: Int) {
  require(key <= rvd.typ.key.length && key >= 0)
  val realType: RVDType = rvd.typ
  val virtType = RVDType(realType.rowType, realType.key.take(key))
  val (kType, _) = rvd.rowType.select(virtType.key)

  private def checkJoinCompatability(right: KeyedRVD) {
    if (!(kType isIsomorphicTo right.kType))
      fatal(
        s"""Incompatible join keys. Keys must have same length and types, in order:
           | Left join key type: ${ kType.toString }
           | Right join key type: ${ right.kType.toString }
         """.stripMargin)
  }

  private def checkLeftIntervalJoinCompatability(right: KeyedRVD) {
    if (!(kType.size == 1 && right.kType.size == 1
      && kType.types(0) == right.kType.types(0).asInstanceOf[TInterval].pointType))
      fatal(
        s"""Incompatible join keys in left interval join:
           | Left join key type: ${ kType.toString }
           | Right join key type: ${ right.kType.toString }
         """.stripMargin)
  }

  def orderedLeftIntervalJoin(
    ctx: ExecuteContext,
    right: KeyedRVD,
    joiner: PStruct => (RVDType, (RVDContext, Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => Iterator[RegionValue])
  ): RVD = {
    checkLeftIntervalJoinCompatability(right)

    val lTyp = virtType
    val rTyp = right.virtType

    rvd.intervalAlignAndZipPartitions(ctx, right.rvd) {
      t: PStruct => {
        val (newTyp, f) = joiner(t)

        (newTyp, (ctx: RVDContext, it: Iterator[RegionValue], intervals: Iterator[RegionValue]) =>
          f(
            ctx,
            OrderedRVIterator(lTyp, it, ctx)
              .leftIntervalJoin(OrderedRVIterator(rTyp, intervals, ctx))))
      }
    }
  }

  def orderedLeftIntervalJoinDistinct(
    ctx: ExecuteContext,
    right: KeyedRVD,
    joiner: PStruct => (RVDType, (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue])
  ): RVD = {
    checkLeftIntervalJoinCompatability(right)

    val lTyp = virtType
    val rTyp = right.virtType

    rvd.intervalAlignAndZipPartitions(ctx, right.rvd) {
      t: PStruct => {
        val (newTyp, f) = joiner(t)

        (newTyp, (ctx: RVDContext, it: Iterator[RegionValue], intervals: Iterator[RegionValue]) =>
          f(
            ctx,
            OrderedRVIterator(lTyp, it, ctx)
              .leftIntervalJoinDistinct(OrderedRVIterator(rTyp, intervals, ctx))))
      }
    }
  }

  def orderedMerge(
    right: KeyedRVD,
    ctx: ExecuteContext
  ): RVD = {
    checkJoinCompatability(right)
    require(this.realType.rowType == right.realType.rowType)

    if (key == 0)
      return RVD.unkeyed(
        this.realType.rowType,
        ContextRDD.union(
          rvd.sparkContext,
          Seq(this.rvd.crdd, right.rvd.crdd)))

    val ranges = this.rvd.partitioner.coarsenedRangeBounds(key) ++
      right.rvd.partitioner.coarsenedRangeBounds(key)
    val newPartitioner = RVDPartitioner.generate(virtType.key, kType, ranges)

    val repartitionedLeft = this.rvd.repartition(ctx, newPartitioner)
    val lType = this.virtType
    val rType = right.virtType

    repartitionedLeft.alignAndZipPartitions(
      this.virtType,
      right.rvd,
      key
    ) { (ctx, leftIt, rightIt) =>
      OrderedRVIterator(lType, leftIt, ctx)
        .merge(OrderedRVIterator(rType, rightIt, ctx))
    }
  }
}
