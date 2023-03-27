package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{RepartitionStrategy, TableAggregateByKey, TableDistinct, TableExplode, TableFilter, TableFilterIntervals, TableGen, TableHead, TableIR, TableIntervalJoin, TableJoin, TableKeyBy, TableLeftJoinRightDistinct, TableLiteral, TableMapGlobals, TableMapPartitions, TableMapRows, TableMultiWayZipJoin, TableOrderBy, TableParallelize, TableRange, TableRead, TableRename, TableRepartition, TableTail, TableToTableApply, TableUnion}
import is.hail.methods.TableFilterPartitions
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual.TStruct
import is.hail.utils._
import org.apache.spark.sql.Row

/*
 *  PlanPartitioning is a query optimization analysis that determines a partitioner to pass down to minimize work
 *  for joins.
 *
 *  we do this by disentangling partitioners from tablestages.
 *  1. we add an analysis that computes partitioners bottom-up
 *  2. we make a decision about partitioners at any root of a TableIR subtree (e.g. Table consumers, unkeyed => keyed)
 *  3. we are then passing down the decision as we lower TableIR.
 *     the structure of LowerTableIR -- called by the root, which recursively lowers its children.
 */

case class PartitionInfo(partitioner: RVDPartitioner, sparsity: Option[IndexedSeq[Int]])

object PartitionProposal {
  def fromPartitioner(partitioner: RVDPartitioner): PartitionProposal =
    PartitionProposal(Some(partitioner), partitioner.allowedOverlap, PlanPartitioning.REPARTITION_REQUIRES_EXTRA_READ)
}


sealed trait RequestedPartitioning {
  def foreach(f: RVDPartitioner => Unit)
}

case object UseTheDefaultPartitioning extends RequestedPartitioning {
  override def foreach(f: RVDPartitioner => Unit): Unit = ()
}
case class UseThisPartitioning(partitioner: RVDPartitioner) extends RequestedPartitioning {
  override def foreach(f: RVDPartitioner => Unit): Unit = f(partitioner)
}


case class PartitionProposal(
  // partitioner is None if we cannot compute static information about key ranges
  partitioner: Option[RVDPartitioner],
  allowedOverlap: Int,
  defaultPartitioningAffinity: Int) {
  require(partitioner.forall(_.kType.size > 0))

  def getPartitioner(requestedAllowedOverlap: Int): Option[RVDPartitioner] = {
    partitioner.map(_.strictify(requestedAllowedOverlap))
  }

  def chooseBest(): RequestedPartitioning = {
    val p = partitioner match {
      case Some(p) => UseThisPartitioning(p)
      case None => UseTheDefaultPartitioning
    }

    p
  }
}

object PlanPartitioning {

  def noInformation(): PartitionProposal = PartitionProposal(None, 0, NO_AFFINITY)

  val NO_AFFINITY = 0
  val REPARTITION_REQUIRES_EXTRA_READ = 1

//  def computeUpperBoundWasteOfIntersection(requested: RVDPartitioner, base: RVDPartitioner): Double = {
//    var partsRead = 0
//    base.rangeBounds.foreach { r =>
//      val (start, end) = requested.intervalRange(r)
//      partsRead += (end - start + 1)
//    }
//    ???
//  }

  def unionPlan(ctx: ExecuteContext, plans: IndexedSeq[PartitionProposal], keyType: TStruct): PartitionProposal = {
    val allowedOverlap = math.min(plans.map(_.allowedOverlap).min, keyType.size - 1)
    if (plans.map(_.partitioner).exists(_.isEmpty))
      return PartitionProposal(None, allowedOverlap, REPARTITION_REQUIRES_EXTRA_READ)

    val parts = plans.map(_.partitioner.get)
    val newPartitioner = RVDPartitioner.generate(ctx.stateManager, keyType, parts.flatMap(_.rangeBounds))
    PartitionProposal(Some(newPartitioner), allowedOverlap, plans.map(_.defaultPartitioningAffinity).max)
  }

  def joinedPlan(ctx: ExecuteContext, left: PartitionProposal, right: PartitionProposal, joinKey: Int, joinType: String, resultKey: TStruct): PartitionProposal = {

    val allowedOverlap = joinKey - 1
    val newPart = left.getPartitioner(allowedOverlap).liftedZip(right.getPartitioner(allowedOverlap)).map { case (leftPart, right) =>
      def rightPart: RVDPartitioner = right.coarsen(joinKey).extendKey(resultKey)

      (joinType: @unchecked) match {
        case "left" => leftPart
        case "right" => rightPart
        case "inner" => leftPart.intersect(rightPart)
        case "outer" => RVDPartitioner.generate(
          ctx.stateManager,
          resultKey.fieldNames.take(joinKey),
          resultKey,
          leftPart.rangeBounds ++ rightPart.rangeBounds)
      }
    }

    PartitionProposal(newPart,
      allowedOverlap = math.min(left.allowedOverlap, joinKey - 1),
      defaultPartitioningAffinity = math.max(left.defaultPartitioningAffinity, right.defaultPartitioningAffinity))
  }

  def analyze(ctx: ExecuteContext, x: TableIR): PartitionProposal = {
    def recur(tir: TableIR): PartitionProposal = analyze(ctx, tir)

    val part = x match {
      case t if t.typ.key.isEmpty =>
        noInformation()
      case TableRead(t, _, tr) =>
        tr.partitionProposal(ctx)
      case t: TableLiteral =>
        PartitionProposal(Some(t.rvd.partitioner), t.rvd.typ.key.length, REPARTITION_REQUIRES_EXTRA_READ)
      case TableRepartition(child, n, RepartitionStrategy.NAIVE_COALESCE) =>
        val pc = recur(child)
        PartitionProposal(pc.partitioner.map(_.naiveCoalesce(n)), pc.allowedOverlap, pc.defaultPartitioningAffinity)
      case tr@TableRange(n, nPartitions) =>
        val ranges = tr.defaultPartitionRanges()
        val part = new RVDPartitioner(ctx.stateManager, tr.typ.keyType,
          ranges.map { case (startIncl, endExcl) => Interval(Row(startIncl), Row(endExcl), true, false) },
          1)
        PartitionProposal(Some(part), 1, NO_AFFINITY)
      case tk@TableKeyBy(child, keys, isSorted) =>
        val pc = recur(child)

        val preservedKey = (child.typ.key, keys).zipped.takeWhile { case (s1, s2) => s1 == s2 }.size
        pc.copy(partitioner = pc.partitioner.map(p => p.coarsen(preservedKey).extendKey(tk.typ.keyType)))
      case TableFilter(child, _) =>
        recur(child)
      case TableHead(child, _) =>
        recur(child)
      case TableTail(child, _) =>
        recur(child)
      case tj@TableJoin(left, right, joinType, joinKey) =>
        joinedPlan(ctx, recur(left), recur(right), joinKey = joinKey, joinType = joinType, resultKey = tj.typ.keyType)
      case TableIntervalJoin(child1, intervalChild, _, _) =>
        // intervalChild will be shuffled to child1 partitioning
      recur(child1)
      case TableLeftJoinRightDistinct(left, right, _) =>
        joinedPlan(ctx, recur(left), recur(right), joinKey = left.typ.keyType.size,
          joinType = "left",
          resultKey = left.typ.keyType)
      case TableMapPartitions(child, _, _, _, _, allowedOverlap) =>
        val pc = recur(child)
        pc.copy(allowedOverlap = math.min(allowedOverlap, pc.allowedOverlap))
      case TableMapRows(child, _) =>
        recur(child)
      case TableMapGlobals(child, _) =>
        recur(child)
      case TableExplode(child, _) =>
        recur(child)
      case tu@TableUnion(tables) =>
        val kt = tu.typ.keyType
        unionPlan(ctx, tables.map(recur), kt)
      case tj@TableMultiWayZipJoin(tables, _, _) =>
        val kt = tj.typ.keyType
        unionPlan(ctx, tables.map(recur), kt)
      case TableDistinct(child) =>
        val pc = recur(child)
        pc.copy(allowedOverlap = math.min(child.typ.key.length - 1, pc.allowedOverlap))
      case TableAggregateByKey(child, _) =>
        val pc = recur(child)
        pc.copy(allowedOverlap = math.min(child.typ.key.length - 1, pc.allowedOverlap))
      case TableRename(child, renameMap, _) =>
        val pc = recur(child)
        pc.copy(partitioner = pc.partitioner.map(_.rename(renameMap)))
      case TableFilterIntervals(child, intervals, keep) =>
        val pc = recur(child)
        pc
      case TableGen(contexts, globals, cname, gname, body, partitioner, errorId) =>
        PartitionProposal(Some(partitioner), partitioner.kType.size, REPARTITION_REQUIRES_EXTRA_READ)
      case TableToTableApply(child, TableFilterPartitions(intervals, keep)) =>
        recur(child)
      case t: TableToTableApply =>
        noInformation()
    }

    part
  }
}
