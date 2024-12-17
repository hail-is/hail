package is.hail.rvd

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.sparkextras._
import is.hail.types.physical.PStruct
import is.hail.types.virtual.TInterval
import is.hail.utils._

import scala.collection.generic.Growable

class KeyedRVD(val rvd: RVD, val key: Int) {
  require(key <= rvd.typ.key.length && key >= 0)
  val realType: RVDType = rvd.typ
  val virtType = RVDType(realType.rowType, realType.key.take(key))
  val (kType, _) = rvd.rowType.select(virtType.key)

  private def checkJoinCompatability(right: KeyedRVD): Unit = {
    if (!(kType isJoinableWith right.kType))
      fatal(
        s"""Incompatible join keys. Keys must have same length and types, in order:
           | Left join key type: ${kType.toString}
           | Right join key type: ${right.kType.toString}
         """.stripMargin
      )
  }

  private def checkLeftIntervalJoinCompatability(right: KeyedRVD): Unit = {
    if (
      !(kType.size == 1 && right.kType.size == 1
        && kType.types(0) == right.kType.types(0).asInstanceOf[TInterval].pointType)
    )
      fatal(
        s"""Incompatible join keys in left interval join:
           | Left join key type: ${kType.toString}
           | Right join key type: ${right.kType.toString}
         """.stripMargin
      )
  }

  // 'joinedType.key' must be the join key, followed by the remaining left key,
  // followed by the (possibly renamed) remaining right key. 'joiner' must copy
  // these 'joinedType.key' fields from the corresponding fields in the
  // JoinedRegionValue.
  def orderedJoin(
    right: KeyedRVD,
    joinType: String,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType,
    ctx: ExecuteContext,
  ): RVD = {
    checkJoinCompatability(right)

    val sm = ctx.stateManager
    val newPartitioner = {
      def leftPart = this.rvd.partitioner.strictify()
      def rightPart = right.rvd.partitioner.coarsen(key).extendKey(realType.kType.virtualType)
      (joinType: @unchecked) match {
        case "left" => leftPart
        case "right" => rightPart
        case "inner" => leftPart.intersect(rightPart)
        case "outer" => RVDPartitioner.generate(
            sm,
            kType.fieldNames,
            realType.kType.virtualType,
            leftPart.rangeBounds ++ rightPart.rangeBounds,
          )
      }
    }
    val repartitionedLeft = rvd.repartition(ctx, newPartitioner)
    val compute
      : (OrderedRVIterator, OrderedRVIterator, Iterable[RegionValue] with Growable[RegionValue]) => Iterator[JoinedRegionValue] =
      (joinType: @unchecked) match {
        case "inner" => _.innerJoin(_, _)
        case "left" => _.leftJoin(_, _)
        case "right" => _.rightJoin(_, _)
        case "outer" => _.outerJoin(_, _)
      }
    val lTyp = virtType
    val rTyp = right.virtType
    val rRowPType = right.realType.rowType

    repartitionedLeft.alignAndZipPartitions(
      joinedType.copy(key = joinedType.key.take(realType.key.length)),
      right.rvd,
      key,
    ) { (ctx, leftIt, rightIt) =>
      val sideBuffer = ctx.freshRegion()
      joiner(
        ctx,
        compute(
          OrderedRVIterator(lTyp, leftIt, ctx, sm),
          OrderedRVIterator(rTyp, rightIt, ctx, sm),
          new RegionValueArrayBuffer(rRowPType, sideBuffer, sm),
        ),
      )
    }.extendKeyPreservesPartitioning(ctx, joinedType.key)
  }

  def orderedLeftIntervalJoin(
    executeContext: ExecuteContext,
    right: KeyedRVD,
    joiner: PStruct => (
      RVDType,
      (RVDContext, Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => Iterator[RegionValue],
    ),
  ): RVD = {
    checkLeftIntervalJoinCompatability(right)

    val lTyp = virtType
    val rTyp = right.virtType

    val sm = executeContext.stateManager
    rvd.intervalAlignAndZipPartitions(executeContext, right.rvd) {
      t: PStruct =>
        val (newTyp, f) = joiner(t)

        (
          newTyp,
          (ctx: RVDContext, it: Iterator[RegionValue], intervals: Iterator[RegionValue]) =>
            f(
              ctx,
              OrderedRVIterator(lTyp, it, ctx, sm)
                .leftIntervalJoin(OrderedRVIterator(rTyp, intervals, ctx, sm)),
            ),
        )
    }
  }

  def orderedLeftIntervalJoinDistinct(
    executeContext: ExecuteContext,
    right: KeyedRVD,
    joiner: PStruct => (RVDType, (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue]),
  ): RVD = {
    checkLeftIntervalJoinCompatability(right)

    val lTyp = virtType
    val rTyp = right.virtType

    val sm = executeContext.stateManager
    rvd.intervalAlignAndZipPartitions(executeContext, right.rvd) {
      t: PStruct =>
        val (newTyp, f) = joiner(t)

        (
          newTyp,
          (ctx: RVDContext, it: Iterator[RegionValue], intervals: Iterator[RegionValue]) =>
            f(
              ctx,
              OrderedRVIterator(lTyp, it, ctx, sm)
                .leftIntervalJoinDistinct(OrderedRVIterator(rTyp, intervals, ctx, sm)),
            ),
        )
    }
  }

  def orderedLeftJoinDistinct(
    right: KeyedRVD,
    joiner: (RVDContext, Iterator[JoinedRegionValue]) => Iterator[RegionValue],
    joinedType: RVDType,
  ): RVD = {
    checkJoinCompatability(right)
    val lTyp = virtType
    val rTyp = right.virtType
    val sm = right.rvd.partitioner.sm

    rvd.alignAndZipPartitions(
      joinedType,
      right.rvd,
      key,
    ) { (ctx, leftIt, rightIt) =>
      joiner(
        ctx,
        OrderedRVIterator(lTyp, leftIt, ctx, sm).leftJoinDistinct(OrderedRVIterator(
          rTyp,
          rightIt,
          ctx,
          sm,
        )),
      )
    }
  }

  def orderedMerge(
    right: KeyedRVD,
    ctx: ExecuteContext,
  ): RVD = {
    checkJoinCompatability(right)
    require(this.realType.rowType == right.realType.rowType)

    if (key == 0)
      return RVD.unkeyed(
        this.realType.rowType,
        ContextRDD.union(
          rvd.sparkContext,
          Seq(this.rvd.crdd, right.rvd.crdd),
        ),
      )

    val ranges = this.rvd.partitioner.coarsenedRangeBounds(key) ++
      right.rvd.partitioner.coarsenedRangeBounds(key)
    val newPartitioner = RVDPartitioner.generate(ctx.stateManager, virtType.key, kType, ranges)

    val repartitionedLeft = this.rvd.repartition(ctx, newPartitioner)
    val lType = this.virtType
    val rType = right.virtType
    val sm = ctx.stateManager

    repartitionedLeft.alignAndZipPartitions(
      this.virtType,
      right.rvd,
      key,
    ) { (ctx, leftIt, rightIt) =>
      OrderedRVIterator(lType, leftIt, ctx, sm)
        .merge(OrderedRVIterator(rType, rightIt, ctx, sm))
    }
  }
}
